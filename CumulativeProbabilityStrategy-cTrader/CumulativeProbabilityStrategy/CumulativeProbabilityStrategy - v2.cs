using cAlgo.API;
using cAlgo.API.Internals;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using static System.Net.WebRequestMethods;
using System.Diagnostics;
using System.Linq;

/*──────────────────────────────────────────────────────────────────────────────┐
│  ENHANCED Cumulative‑Probability Strategy – PF 1.46 → 2.0+ optimizations     │
│  =============================================================================│
│  NOWE FUNKCJE:                                                                │
│  • Confidence-based filtering (min confidence threshold)                      │
│  • Dynamic position sizing (Kelly criterion + confidence)                     │
│  • Next-step prediction integration (direction boost)                         │
│  • Volatility-adjusted targets (adaptive TP/SL)                               │
│  • Time-based filtering (avoid weak hours)                                    │
│  • Advanced risk management (daily limits, max positions)                     │
│  • Market regime detection (H4 trend confirmation)                            │
└──────────────────────────────────────────────────────────────────────────────*/

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class EnhancedCumulativeProbabilityStrategy : Robot
    {
        /*──────── Parametry użytkownika ───────────────────────────────────*/
        [Parameter("API URL", DefaultValue = "http://localhost:5051/predict")] public string ApiUrl { get; set; }
        [Parameter("Sequence Length", DefaultValue = 210)] public int SeqLen { get; set; }
        [Parameter("Lookahead", DefaultValue = 1)] public int Lookahead { get; set; }

        // ENHANCED: Risk Management
        [Parameter("Base Volume", DefaultValue = 1000)] public double BaseVolume { get; set; }
        [Parameter("Max Volume Multiplier", DefaultValue = 3.0)] public double MaxVolMult { get; set; }
        [Parameter("Commission (pips)", DefaultValue = 0.3)] public double CommissionPips { get; set; }

        // ENHANCED: Confidence & Quality Filters
        [Parameter("Min Confidence", DefaultValue = 0.6)] public double MinConfidence { get; set; }
        [Parameter("Min EV Threshold", DefaultValue = 1.0)] public double MinEVThreshold { get; set; }
        [Parameter("Min Predicted Move %", DefaultValue = 0.5)] public double MinPredictedMove { get; set; }

        // ENHANCED: Advanced Risk Management
        [Parameter("Max Daily Risk %", DefaultValue = 2.0)] public double MaxDailyRisk { get; set; }
        [Parameter("Max Concurrent Positions", DefaultValue = 3)] public int MaxPositions { get; set; }
        [Parameter("Kelly Fraction Cap", DefaultValue = 0.25)] public double KellyFractionCap { get; set; }

        // ENHANCED: Time & Market Filters
        [Parameter("Enable Time Filter", DefaultValue = true)] public bool EnableTimeFilter { get; set; }
        [Parameter("Enable Trend Filter", DefaultValue = true)] public bool EnableTrendFilter { get; set; }
        [Parameter("H4 Trend Lookback", DefaultValue = 24)] public int H4TrendLookback { get; set; }

        /*──────── Stałe wewnętrzne ───────────────────────────────────────*/
        private const int MaxPips = 25; // Zwiększone dla większych targetów
        private const int EntryOffsetPips = 1;
        private int BarsNeeded => SeqLen + Lookahead;
        private HttpClient _http;

        // ENHANCED: Performance tracking
        private double _dailyStartBalance;
        private DateTime _lastResetDate;

        /*──────── Pomocniczy rekord setupu ───────────────────────────────*/
        private record Setup(int TpDist, int SlDist, int entryDist, double EV);
        private record EnhancedPrediction(
            Dictionary<string, Dictionary<string, double>> Thresholds,
            double PredictedCloseChange,
            string Direction,
            double AvgConfidence,
            double MaxConfidence,
            double PredictedVolatility
        );

        /*══════════════════ 1. Start & OnBar ═════════════════════════════*/
        protected override void OnStart()
        {
            _http = new HttpClient(new HttpClientHandler { Proxy = null, UseProxy = false }, disposeHandler: true);
            _dailyStartBalance = Account.Balance;
            _lastResetDate = Server.Time.Date;

            Print("🚀 ENHANCED Strategy Started!");
            Print($"   Min Confidence: {MinConfidence:F2}");
            Print($"   Min EV: {MinEVThreshold:F2}");
            Print($"   Max Daily Risk: {MaxDailyRisk:F1}%");
            Print($"   Kelly Cap: {KellyFractionCap:F2}");
        }

        protected override void OnBar()
        {
            // Reset daily tracking
            if (Server.Time.Date != _lastResetDate)
            {
                _dailyStartBalance = Account.Balance;
                _lastResetDate = Server.Time.Date;
                Print($"📅 Daily reset: Starting balance {_dailyStartBalance:F2}");
            }

            int iBar = Bars.Count - 1;
            if (iBar < BarsNeeded) return;

            // ENHANCED: Advanced risk checks
            if (!PassesRiskChecks()) return;
            if (!PassesTimeFilter()) return;

            var prediction = FetchEnhancedPrediction(iBar);
            if (prediction == null) return;

            // ENHANCED: Confidence filtering
            if (prediction.MaxConfidence < MinConfidence)
            {
                Print($"⚠️ Low confidence: {prediction.MaxConfidence:F3} < {MinConfidence:F2} - skipping");
                return;
            }

            // ENHANCED: Predicted move filtering
            if (Math.Abs(prediction.PredictedCloseChange) < MinPredictedMove)
            {
                Print($"⚠️ Small predicted move: {prediction.PredictedCloseChange:F2}% < {MinPredictedMove:F1}% - skipping");
                return;
            }

            if (Positions.Any(p => p.StopLoss == null || p.TakeProfit == null))
            {
                UpdateAllMissingSLTP();
            }

            //ManageOpenPositions(iBar, prediction);

            //if (Positions.Count < MaxPositions)
            {
                var (bestL, bestS) = FindEnhancedEntrySetups(prediction);

                if (bestL.EV > MinEVThreshold)
                {
                    Print($"🟢 LONG setup: EV={bestL.EV:F2}, Conf={prediction.MaxConfidence:F3}");
                    SubmitEnhancedLimit(iBar, true, bestL, prediction);
                }
                if (bestS.EV > MinEVThreshold)
                {
                    Print($"🔴 SHORT setup: EV={bestS.EV:F2}, Conf={prediction.MaxConfidence:F3}");
                    SubmitEnhancedLimit(iBar, false, bestS, prediction);
                }
            }
        }

        /*══════════════════ 2. ENHANCED Pobieranie predykcji ═══════════*/
        private EnhancedPrediction FetchEnhancedPrediction(int idx)
        {
            var h1Data = new List<double[]>();
            var h1Timestamps = new List<string>();

            for (int i = 0; i < BarsNeeded; i++)
            {
                int b = idx - BarsNeeded + i + 1;
                h1Data.Add(new double[] {
                    Bars.OpenPrices[b], Bars.HighPrices[b], Bars.LowPrices[b],
                    Bars.ClosePrices[b], Bars.TickVolumes[b]
                });
                h1Timestamps.Add(Bars.OpenTimes[b].ToString("yyyy-MM-ddTHH:mm:ss.fffZ"));
            }

            var requestData = new { h1_data = h1Data, h1_timestamps = h1Timestamps };
            string json = JsonConvert.SerializeObject(requestData);

            try
            {
                var resp = _http.PostAsync(ApiUrl, new StringContent(json, Encoding.UTF8, "application/json")).Result;
                var responseText = resp.Content.ReadAsStringAsync().Result;

                if (!resp.IsSuccessStatusCode)
                {
                    Print($"API error: {resp.StatusCode} - {responseText}");
                    return null;
                }

                var fullResponse = JsonConvert.DeserializeObject<dynamic>(responseText);

                // Parse enhanced data
                var thresholds = new Dictionary<string, Dictionary<string, double>>();
                thresholds["up"] = JsonConvert.DeserializeObject<Dictionary<string, double>>(fullResponse.up.ToString());
                thresholds["down"] = JsonConvert.DeserializeObject<Dictionary<string, double>>(fullResponse.down.ToString());

                var nextStep = fullResponse.next_step_prediction.summary;
                var analytics = fullResponse.advanced_analytics.confidence_metrics;

                return new EnhancedPrediction(
                    thresholds,
                    (double)nextStep.predicted_close_change_pct,
                    (string)nextStep.overall_direction,
                    (double)analytics.avg_threshold_confidence,
                    (double)analytics.max_threshold_confidence,
                    (double)nextStep.predicted_volatility
                );
            }
            catch (Exception ex)
            {
                Print($"API error: {ex.Message}");
                return null;
            }
        }

        /*══════════════════ 3. ENHANCED Risk Checks ═════════════════════*/
        private bool PassesRiskChecks()
        {
            // Daily risk limit
            double dailyPnL = Account.Balance - _dailyStartBalance;
            double dailyRiskPct = Math.Abs(dailyPnL) / _dailyStartBalance * 100;

            if (dailyRiskPct > MaxDailyRisk)
            {
                Print($"🛑 Daily risk limit exceeded: {dailyRiskPct:F1}% > {MaxDailyRisk:F1}%");
                return false;
            }

            // Max positions limit
            if (Positions.Count >= MaxPositions)
            {
                Print($"🛑 Max positions limit: {Positions.Count} >= {MaxPositions}");
                return false;
            }

            return true;
        }

        private bool PassesTimeFilter()
        {
            if (!EnableTimeFilter) return true;

            DateTime utcTime = Server.Time;
            int hour = utcTime.Hour;

            // Good trading hours (UTC):
            // 08:00-17:00 (London/NY overlap)
            // 01:00-03:00 (Asian close volatility)
            bool isGoodTime = (hour >= 8 && hour <= 17) || (hour >= 1 && hour <= 3);

            if (!isGoodTime)
            {
                Print($"⏰ Outside trading hours: {hour}:00 UTC - skipping");
                return false;
            }

            return true;
        }

        /*══════════════════ 4. ENHANCED Setup Finding ═══════════════════*/
        private (Setup, Setup) FindEnhancedEntrySetups(EnhancedPrediction prediction)
        {
            double pip = Symbol.PipSize;
            double fee = Symbol.Spread / pip / 2.0 + CommissionPips;
            Setup bestL = new(0, 0, 0, double.NegativeInfinity);
            Setup bestS = new(0, 0, 0, double.NegativeInfinity);

            var probs = prediction.Thresholds;
            var upKeys = probs["up"].Keys.ToList();
            var downKeys = probs["down"].Keys.ToList();

            // Add ≥0pips = 1.0
            if (!probs["up"].ContainsKey("≥0pips")) probs["up"]["≥0pips"] = 1.0;
            if (!probs["down"].ContainsKey("≥0pips")) probs["down"]["≥0pips"] = 1.0;

            int setupsChecked = 0;

            foreach (var entryKey in upKeys.Concat(downKeys).Distinct())
            {
                if (!TryExtractPipsFromKey(entryKey, out double entryPips)) continue;
                if (entryPips <= 0 || entryPips > MaxPips) continue;

                int thrEntry = (int)Math.Round(entryPips);

                foreach (var slKey in upKeys.Concat(downKeys).Distinct())
                {
                    if (!TryExtractPipsFromKey(slKey, out double slPips)) continue;
                    int thrSL = (int)Math.Round(slPips);
                    if (thrSL <= thrEntry + 1) continue; // Minimum 1 pip difference
                    if (thrSL > MaxPips) continue;

                    foreach (var tpKey in upKeys.Concat(downKeys).Distinct())
                    {
                        if (!TryExtractPipsFromKey(tpKey, out double tpPips)) continue;
                        int thrTP = (int)Math.Round(tpPips);
                        if (thrTP < 1) continue;
                        if (thrTP > MaxPips) continue;

                        setupsChecked++;

                        if (!probs["down"].ContainsKey(entryKey) || !probs["up"].ContainsKey(entryKey)) continue;
                        if (!probs["down"].ContainsKey(slKey) || !probs["up"].ContainsKey(slKey)) continue;
                        if (!probs["up"].ContainsKey(tpKey) || !probs["down"].ContainsKey(tpKey)) continue;

                        double eL = probs["down"][entryKey], eS = probs["up"][entryKey];
                        double lL = probs["down"][slKey], lS = probs["up"][slKey];
                        double pL = probs["up"][tpKey], pS = probs["down"][tpKey];

                        if (pL > 0 && eL > 0)
                        {
                            var newL = EvalEnhancedBranch(eL, pL, lL, thrEntry, thrSL, thrTP, fee, bestL, prediction, true);
                            if (newL.EV > bestL.EV) bestL = newL;
                        }
                        if (pS > 0 && eS > 0)
                        {
                            var newS = EvalEnhancedBranch(eS, pS, lS, thrEntry, thrSL, thrTP, fee, bestS, prediction, false);
                            if (newS.EV > bestS.EV) bestS = newS;
                        }
                    }
                }
            }

            Print($"📊 Enhanced scan: {setupsChecked} setups, Best L={bestL.EV:F2}, Best S={bestS.EV:F2}");
            return (bestL, bestS);
        }

        private Setup EvalEnhancedBranch(double pEntry, double pProfit, double pLoss,
                                       int distEntry, int distSL, int distTP, double fee,
                                       Setup currentBest, EnhancedPrediction prediction, bool isLong)
        {
            // Basic EV calculation
            double pLossGivenEntry = pLoss / pEntry;
            double pProfitGivenEntry = pProfit / pEntry;
            double basicEV = pProfitGivenEntry * (distEntry + distTP) - pLossGivenEntry * (distSL - distEntry);

            // ENHANCED: Apply multipliers
            double enhancedEV = basicEV;

            // 1. Confidence boost
            double confidenceBoost = 1.0 + (prediction.MaxConfidence - 0.5) * 0.5; // 0.5→1.0, 1.0→1.25
            enhancedEV *= confidenceBoost;

            // 2. Direction alignment boost
            bool directionAligned = (isLong && prediction.Direction == "bullish") ||
                                  (!isLong && prediction.Direction == "bearish");
            if (directionAligned)
            {
                double moveStrength = Math.Abs(prediction.PredictedCloseChange) / 100.0;
                double directionBoost = 1.0 + Math.Min(0.5, moveStrength * 10); // Max 50% boost
                enhancedEV *= directionBoost;
            }
            else
            {
                // Penalize against predicted direction
                enhancedEV *= 0.7;
            }

            // 3. H4 trend confirmation (if enabled)
            if (EnableTrendFilter)
            {
                double h4Trend = GetH4Trend();
                bool trendAligned = (isLong && h4Trend > 5) || (!isLong && h4Trend < -5);
                if (trendAligned) enhancedEV *= 1.2;
                else if ((isLong && h4Trend < -10) || (!isLong && h4Trend > 10)) enhancedEV *= 0.6;
            }

            // 4. Volatility adjustment
            if (prediction.PredictedVolatility > 0.8) enhancedEV *= 1.1; // High vol boost
            if (prediction.PredictedVolatility < 0.3) enhancedEV *= 0.9; // Low vol penalty

            if (enhancedEV > currentBest.EV)
            {
                return new Setup(distTP, distSL, distEntry, enhancedEV);
            }
            return currentBest;
        }

        private double GetH4Trend()
        {
            if (Bars.Count < H4TrendLookback) return 0;

            double currentPrice = Bars.ClosePrices.Last();
            double pastPrice = Bars.ClosePrices[Bars.Count - H4TrendLookback];
            return (currentPrice - pastPrice) / Symbol.PipSize;
        }

        /*══════════════════ 5. ENHANCED Limit Orders ═══════════════════*/
        private void SubmitEnhancedLimit(int idx, bool longSide, Setup s, EnhancedPrediction prediction)
        {
            string labelPref = longSide ? "EV_BUY" : "EV_SELL";

            // Cancel existing orders
            foreach (var po in PendingOrders)
                if (po.Label.StartsWith(labelPref)) CancelPendingOrder(po);

            double pip = Symbol.PipSize;
            double close = Bars.ClosePrices[idx];
            double entryPrice = longSide ? close - s.entryDist * pip : close + s.entryDist * pip;

            // ENHANCED: Volatility-adjusted targets
            int baseTpPips = s.TpDist + s.entryDist;
            int baseSlPips = Math.Max(1, s.SlDist - s.entryDist);

            var (adjustedTpPips, adjustedSlPips) = AdjustTargetsForVolatility(baseTpPips, baseSlPips, prediction.PredictedVolatility);

            // ENHANCED: Dynamic position sizing
            double volume = CalculateOptimalVolume(s.EV, prediction, adjustedSlPips);

            Print($"📈 {labelPref}: Entry={entryPrice:F5}, TP={adjustedTpPips}pips, SL={adjustedSlPips}pips, Vol={volume:F0}");
            Print($"   EV={s.EV:F2}, Conf={prediction.MaxConfidence:F3}, PredMove={prediction.PredictedCloseChange:F2}%");

            var tr = PlaceLimitOrder(
                longSide ? TradeType.Buy : TradeType.Sell,
                SymbolName,
                volume,
                entryPrice,
                $"{labelPref}_C{prediction.MaxConfidence:F2}_EV{s.EV:F1}",
                adjustedSlPips,
                adjustedTpPips
            );

            if (!tr.IsSuccessful)
            {
                Print($"❌ Order failed: {tr.Error}");
            }
        }

        private (int tpPips, int slPips) AdjustTargetsForVolatility(int baseTp, int baseSl, double predictedVol)
        {
            double volMultiplier = 1.0;

            if (predictedVol > 0.8) volMultiplier = 1.3;      // High vol: wider targets
            else if (predictedVol > 0.5) volMultiplier = 1.1;  // Medium vol: slightly wider
            else if (predictedVol < 0.3) volMultiplier = 0.8;  // Low vol: tighter targets

            int adjustedTp = Math.Max(1, (int)(baseTp * volMultiplier));
            int adjustedSl = Math.Max(1, (int)(baseSl * Math.Sqrt(volMultiplier))); // Less aggressive SL adjustment

            return (adjustedTp, adjustedSl);
        }

        private double CalculateOptimalVolume(double ev, EnhancedPrediction prediction, int slPips)
        {
            // Base volume from confidence
            double confidenceMultiplier = Math.Min(MaxVolMult, prediction.MaxConfidence * 2);
            double baseVol = BaseVolume * confidenceMultiplier;

            // Kelly criterion adjustment
            double winRate = prediction.AvgConfidence;
            double avgWin = ev > 0 ? ev : 1;
            double avgLoss = slPips;

            double kellyFraction = (winRate * avgWin - (1 - winRate) * avgLoss) / avgWin;
            kellyFraction = Math.Max(0, Math.Min(KellyFractionCap, kellyFraction));

            // Risk-based volume
            double riskAmount = Account.Balance * kellyFraction;
            double riskBasedVolume = riskAmount / (slPips * Symbol.PipValue);

            // Use smaller of confidence-based or risk-based volume
            double finalVolume = Math.Min(baseVol, riskBasedVolume);

            return Symbol.NormalizeVolumeInUnits((long)finalVolume, RoundingMode.Down);
        }

        /*══════════════════ 6. Helper Methods ═══════════════════════════*/
        private bool TryExtractPipsFromKey(string key, out double pips)
        {
            pips = 0;
            if (string.IsNullOrEmpty(key) || !key.StartsWith("≥") || !key.EndsWith("pips"))
                return false;

            string pipsStr = key.Substring(1, key.Length - 5);
            return double.TryParse(pipsStr, System.Globalization.NumberStyles.Float,
                                 System.Globalization.CultureInfo.InvariantCulture, out pips);
        }

        private static (int kBest, double evBest) BestContinuationK(Dictionary<string, double> pCum, double moved, double fee)
        {
            int shift = (int)Math.Round(moved);
            int bestK = 0;
            double bestEV = double.NegativeInfinity;

            foreach (var kvp in pCum)
            {
                if (!kvp.Key.StartsWith("≥") || !kvp.Key.EndsWith("pips")) continue;

                string pipsStr = kvp.Key.Substring(1, kvp.Key.Length - 5);
                if (!double.TryParse(pipsStr, System.Globalization.NumberStyles.Float,
                                   System.Globalization.CultureInfo.InvariantCulture, out double pips)) continue;

                int k = (int)Math.Round(pips) - shift;
                if (k <= 0) continue;

                double pTPcum = kvp.Value;
                double gain = k - fee;
                double loss = 1 + fee;
                double ev = pTPcum * gain - (1 - pTPcum) * loss;

                if (ev > bestEV) { bestEV = ev; bestK = k; }
            }
            return (bestK, bestEV);
        }

        /*══════════════════ 7. Position Management ═════════════════════*/
        private void UpdateAllMissingSLTP()
        {
            foreach (var position in Positions)
            {
                if (!position.StopLoss.HasValue || !position.TakeProfit.HasValue)
                {
                    var openOrder = HistoricalOrders
                        .FirstOrDefault(o => o.PositionId == position.Id && o.Status == HistoricalOrderStatus.Filled);

                    if (openOrder == null) continue;

                    if (!position.StopLoss.HasValue && openOrder.StopLoss.HasValue)
                        TryUpdateSL(position, openOrder.StopLoss.Value);

                    if (!position.TakeProfit.HasValue && openOrder.TakeProfit.HasValue)
                        TryUpdateTP(position, openOrder.TakeProfit.Value);
                }
            }
        }

        private void TryUpdateSL(Position position, double targetSL)
        {
            double sl = targetSL;
            bool isLong = position.TradeType == TradeType.Buy;

            // Close if SL already hit
            if ((Bars.LastBar.Close > targetSL && !isLong) || (Bars.LastBar.Close < targetSL && isLong))
            {
                position.Close();
                return;
            }

            for (int attempts = 0; attempts < 20; attempts++)
            {
                var result = position.ModifyStopLossPrice(sl);
                if (result.IsSuccessful) return;

                sl += isLong ? -Symbol.PipSize : Symbol.PipSize;
            }
        }

        private void TryUpdateTP(Position position, double targetTP)
        {
            double tp = targetTP;
            bool isLong = position.TradeType == TradeType.Buy;

            // Close if TP already hit
            if ((Bars.LastBar.Close < targetTP && !isLong) || (Bars.LastBar.Close > targetTP && isLong))
            {
                position.Close();
                return;
            }

            for (int attempts = 0; attempts < 20; attempts++)
            {
                var result = position.ModifyTakeProfitPrice(tp);
                if (result.IsSuccessful) return;

                tp += isLong ? Symbol.PipSize : -Symbol.PipSize;
            }
        }
    }
}
