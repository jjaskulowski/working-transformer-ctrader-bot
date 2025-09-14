//using cAlgo.API;
//using cAlgo.API.Internals;
//using System;
//using System.Collections.Generic;
//using System.Net.Http;
//using System.Text;
//using System.Threading.Tasks;
//using Newtonsoft.Json;
//using static System.Net.WebRequestMethods;
//using System.Diagnostics;
//using System.Linq;

///*──────────────────────────────────────────────────────────────────────────────┐
//│  Cumulative‑Probability Strategy – czytelna i zgodna z drzewem użytkownika     │
//│  =============================================================================│
//│  • Drzewo zdarzeń (na jeden bar):                                             │
//│        BarStart                                                               │
//│        ├── brak wejścia  – p = 1 − pEntry                                     │
//│        └── ENTRY (offset = 1 pip) – p = pEntry                                │
//│             ├── SL   (−1 pip)                         p = pSL|Entry           │
//│             └── ¬SL  (czyli ruch ≥0)                                          │
//│                 ├── TP  (+d pip)                    p = pTP|¬SL,Entry         │
//│                 └── CloseBar (ani TP, ani SL)        reszta                   │
//│                                                                                │
//│  • EV = pEntry × [ pTP|¬SL × gain  −  pSL|Entry × loss ]                      │
//│      – gain  = d − cost;   loss = 1 + cost                                    │
//│  • pTP|¬SL,Entry oraz pSL|Entry przybliżamy formułą „gambler’s ruin”:         │
//│        pTP|¬SL,Entry = d / (d + 1)      (d – odległość TP w pipach)           │
//│        pSL|Entry    = 1 / (d + 1)                                             │
//│  • ENTRY zawsze 1 pip od ceny (bliżej).                                       │
//└──────────────────────────────────────────────────────────────────────────────*/

//namespace cAlgo.Robots
//{
//    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
//    public class CumulativeProbabilityStrategy : Robot
//    {
//        /*──────── Parametry użytkownika ───────────────────────────────────*/
//        [Parameter("API URL", DefaultValue = "http://localhost:5051/predict")] public string ApiUrl { get; set; }
//        [Parameter("Sequence Length", DefaultValue = 210)] public int SeqLen { get; set; }
//        [Parameter("Lookahead", DefaultValue = 1)] public int Lookahead { get; set; }
//        [Parameter("Volume Units", DefaultValue = 1000)] public double VolumeUnits { get; set; }
//        [Parameter("Commission (pips)", DefaultValue = 0.3)] public double CommissionPips { get; set; }

//        /*──────── Stałe wewnętrzne ───────────────────────────────────────*/
//        private const int MaxPips = 20; // tablica modelu 1…20
//        private const int EntryOffsetPips = 1;  // ENTRY zawsze 1 pip od ceny
//        private int BarsNeeded => SeqLen + Lookahead;
//        private HttpClient _http;

//        /*──────── Pomocniczy rekord setupu ───────────────────────────────*/
//        private record Setup(int TpDist, int SlDist, int entryDist, double EV);

//        /*══════════════════ 1. Start & OnBar ═════════════════════════════*/
//        protected override void OnStart()
//        {
//            _http = new HttpClient(new HttpClientHandler { Proxy = null, UseProxy = false }, disposeHandler: true);
//        }

//        protected override void OnBar()
//        {
//            int iBar = Bars.Count - 1;
//            if (iBar < BarsNeeded) return;

//            var probs = FetchProbabilities(iBar);
//            if (probs == null) return;

//            if (Positions.Any(p => p.StopLoss == null || p.TakeProfit == null))
//            {
//                UpdateAllMissingSLTP();
//            }

//            //ManageOpenPositions(iBar, probs);

//            //if (Positions.Count == 0)
//            {
//                var (bestL, bestS) = FindBestEntrySetups(probs);
//                if (bestL.EV > 0) SubmitLimit(iBar, true, bestL);
//                if (bestS.EV > 0) SubmitLimit(iBar, false, bestS);
//            }
//        }

//        /*══════════════════ 2. Pobieranie prawdopodobieństw ═════════════*/
//        private Dictionary<string, Dictionary<string, double>> FetchProbabilities(int idx)
//        {
//            // Przygotuj dane H1 w formacie zgodnym z nowym serwerem
//            var h1Data = new List<double[]>();
//            var h1Timestamps = new List<string>();

//            for (int i = 0; i < BarsNeeded; i++)
//            {
//                int b = idx - BarsNeeded + i + 1;
//                h1Data.Add(new double[] {
//                    Bars.OpenPrices[b],
//                    Bars.HighPrices[b],
//                    Bars.LowPrices[b],
//                    Bars.ClosePrices[b],
//                    Bars.TickVolumes[b]
//                });
//                h1Timestamps.Add(Bars.OpenTimes[b].ToString("yyyy-MM-ddTHH:mm:ss.fffZ"));
//            }

//            // Nowy format JSON dla dual-output serwera
//            var requestData = new
//            {
//                h1_data = h1Data,
//                h1_timestamps = h1Timestamps
//            };

//            string json = JsonConvert.SerializeObject(requestData);
//            Print($"DEBUG: Sending JSON: {json.Substring(0, Math.Min(200, json.Length))}...");

//            try
//            {
//                var resp = _http.PostAsync(ApiUrl, new StringContent(json, Encoding.UTF8, "application/json")).Result;
//                var responseText = resp.Content.ReadAsStringAsync().Result;

//                Print($"DEBUG: Response status: {resp.StatusCode}");
//                Print($"DEBUG: Response: {responseText.Substring(0, Math.Min(500, responseText.Length))}...");

//                if (!resp.IsSuccessStatusCode)
//                {
//                    Print($"API error: {resp.StatusCode} - {responseText}");
//                    return null;
//                }

//                // Parsuj odpowiedź z dual-output serwera
//                var fullResponse = JsonConvert.DeserializeObject<dynamic>(responseText);

//                // Wyciągnij podstawowe prawdopodobieństwa (backward compatibility)
//                var result = new Dictionary<string, Dictionary<string, double>>();
//                result["up"] = JsonConvert.DeserializeObject<Dictionary<string, double>>(fullResponse.up.ToString());
//                result["down"] = JsonConvert.DeserializeObject<Dictionary<string, double>>(fullResponse.down.ToString());

//                // OPCJONALNIE: Loguj dodatkowe informacje z dual-output
//                try
//                {
//                    if (fullResponse.next_step_prediction != null)
//                    {
//                        var nextStep = fullResponse.next_step_prediction.summary;
//                        Print($"Next step prediction: {nextStep.predicted_close_change_pct}% ({nextStep.overall_direction})");
//                    }

//                    if (fullResponse.advanced_analytics != null)
//                    {
//                        var analytics = fullResponse.advanced_analytics.confidence_metrics;
//                        Print($"Confidence: avg={analytics.avg_threshold_confidence:F3}, max={analytics.max_threshold_confidence:F3}");
//                    }
//                }
//                catch (Exception logEx)
//                {
//                    Print($"Warning: Could not parse additional info: {logEx.Message}");
//                }

//                return result;
//            }
//            catch (Exception ex)
//            {
//                Print($"API error: {ex.Message}");
//                if (ex.InnerException != null)
//                    Print($"Inner exception: {ex.InnerException.Message}");
//                return null;
//            }
//        }

//        /*══════════════════ 3. Zarządzanie otwartymi pozycjami ═══════════*/
//        private void ManageOpenPositions(int idx, Dictionary<string, Dictionary<string, double>> probs)
//        {
//            double pip = Symbol.PipSize;
//            double fee = Symbol.Spread / pip / 2.0 + CommissionPips; // koszt wyjścia
//            double price = Bars.ClosePrices[idx];

//            foreach (var pos in GetMyPositions())
//            {
//                bool longSide = pos.TradeType == TradeType.Buy;
//                string key = longSide ? "up" : "down";

//                double moved = longSide ? (price - pos.EntryPrice) / pip
//                                         : (pos.EntryPrice - price) / pip;
//                if (moved < 0) moved = 0;

//                var (k, ev) = BestContinuationK(probs[key], moved, fee);
//                if (ev <= 0) { ClosePosition(pos); continue; }

//                double newTP = longSide ? pos.EntryPrice + k * pip
//                                         : pos.EntryPrice - k * pip;
//                double newSL = (double)pos.StopLoss;
//                if (moved >= 1)
//                    newSL = longSide ? Math.Max((double)pos.EntryPrice, (double)pos.StopLoss)
//                                     : Math.Min((double)pos.EntryPrice, (double)pos.StopLoss);
//                ModifyPosition(pos, newSL, newTP);
//            }
//        }

//        private IEnumerable<Position> GetMyPositions()
//        {
//            foreach (var p in Positions)
//                if (p.Label.StartsWith("EV_")) yield return p;
//        }

//        private static (int kBest, double evBest) BestContinuationK(Dictionary<string, double> pCum, double moved, double fee)
//        {
//            int shift = (int)Math.Round(moved); int bestK = 0; double bestEV = double.NegativeInfinity;
//            for (int k = 1; k <= MaxPips - shift; k++)
//            {
//                double pTPcum = pCum[$"≥{k + shift}pips"]; // ≈ P(tp)
//                // Gambler: pTP|¬SL = k/(k+1); pSL|Entry = 1/(k+1)
//                double gain = k - fee; double loss = 1 + fee;
//                double ev = pTPcum * gain - (1 - pTPcum) * loss;  // uproszczona wersja
//                if (ev > bestEV) { bestEV = ev; bestK = k; }
//            }
//            return (bestK, bestEV);
//        }

//        /*══════════════════ 4. Skanowanie nowych setupów ═════════════════*/
//        private (Setup, Setup) FindBestEntrySetups(Dictionary<string, Dictionary<string, double>> probs)
//        {
//            double pip = Symbol.PipSize; double fee = Symbol.Spread / pip / 2.0 + CommissionPips;
//            Setup bestL = new(0, 0, 0, double.NegativeInfinity);
//            Setup bestS = new(0, 0, 0, double.NegativeInfinity);
//            double minSlDist = Symbol.MinStopLossDistance;
//            double minTpDist = Symbol.MinTakeProfitDistance;

//            Print($"=== SETUP SCANNING DEBUG ===");
//            Print($"Fee: {fee:F2} pips, MinSL: {minSlDist:F1}, MinTP: {minTpDist:F1}");

//            // Wyciągnij dostępne progi z odpowiedzi serwera
//            var upKeys = probs["up"].Keys.ToList();
//            var downKeys = probs["down"].Keys.ToList();

//            Print($"Available thresholds: UP={upKeys.Count}, DOWN={downKeys.Count}");
//            Print($"UP keys: {string.Join(", ", upKeys)}");
//            Print($"DOWN keys: {string.Join(", ", downKeys)}");
//            Print($"Sample UP probs: {string.Join(", ", probs["up"].Take(3).Select(kv => $"{kv.Key}={kv.Value:F3}"))}");
//            Print($"Sample DOWN probs: {string.Join(", ", probs["down"].Take(3).Select(kv => $"{kv.Key}={kv.Value:F3}"))}");

//            // Dodaj ≥0pips = 1.0 dla kompletności
//            if (!probs["up"].ContainsKey("≥0pips")) probs["up"]["≥0pips"] = 1.0;
//            if (!probs["down"].ContainsKey("≥0pips")) probs["down"]["≥0pips"] = 1.0;

//            int setupsChecked = 0;
//            int validSetups = 0;

//            // Iteruj po dostępnych progach zamiast zakładać stałe wartości
//            foreach (var entryKey in upKeys.Concat(downKeys).Distinct())
//            {
//                // Wyciągnij wartość pips z klucza (np. "≥1.4pips" → 1.4)
//                if (!TryExtractPipsFromKey(entryKey, out double entryPips))
//                {
//                    Print($"Failed to parse entry key: {entryKey}");
//                    continue;
//                }
//                if (entryPips <= 0 || entryPips > MaxPips) continue;

//                int thrEntry = (int)Math.Round(entryPips);

//                foreach (var slKey in upKeys.Concat(downKeys).Distinct())
//                {
//                    if (!TryExtractPipsFromKey(slKey, out double slPips)) continue;
//                    int thrSL = (int)Math.Round(slPips);
//                    if (thrSL <= thrEntry + minSlDist) continue;
//                    if (thrSL > MaxPips) continue;

//                    foreach (var tpKey in upKeys.Concat(downKeys).Distinct())
//                    {
//                        if (!TryExtractPipsFromKey(tpKey, out double tpPips)) continue;
//                        int thrTP = (int)Math.Round(tpPips);
//                        if (thrTP < Math.Max(1, minTpDist - thrEntry)) continue;
//                        if (thrTP > MaxPips) continue;

//                        setupsChecked++;

//                        // Użyj oryginalnych kluczy zamiast formatować na nowo
//                        if (!probs["down"].ContainsKey(entryKey) || !probs["up"].ContainsKey(entryKey)) continue;
//                        if (!probs["down"].ContainsKey(slKey) || !probs["up"].ContainsKey(slKey)) continue;
//                        if (!probs["up"].ContainsKey(tpKey) || !probs["down"].ContainsKey(tpKey)) continue;

//                        double eL = probs["down"][entryKey], eS = probs["up"][entryKey];
//                        double lL = probs["down"][slKey], lS = probs["up"][slKey];
//                        double pL = probs["up"][tpKey], pS = probs["down"][tpKey];

//                        validSetups++;

//                        if (setupsChecked <= 5) // Debug pierwszych 5 setupów
//                        {
//                            Print($"Setup {setupsChecked}: E={thrEntry}, SL={thrSL}, TP={thrTP}");
//                            Print($"  LONG: eL={eL:F3}, pL={pL:F3}, lL={lL:F3}");
//                            Print($"  SHORT: eS={eS:F3}, pS={pS:F3}, lS={lS:F3}");
//                        }

//                        if (pL > 0 && eL > 0)
//                        {
//                            var newL = EvalBranch(eL, pL, lL, thrEntry, thrSL, thrTP, fee, bestL);
//                            if (newL.EV > bestL.EV)
//                            {
//                                Print($"New best LONG: EV={newL.EV:F4} (E={thrEntry}, SL={thrSL}, TP={thrTP})");
//                                bestL = newL;
//                            }
//                        }
//                        if (pS > 0 && eS > 0)
//                        {
//                            var newS = EvalBranch(eS, pS, lS, thrEntry, thrSL, thrTP, fee, bestS);
//                            if (newS.EV > bestS.EV)
//                            {
//                                Print($"New best SHORT: EV={newS.EV:F4} (E={thrEntry}, SL={thrSL}, TP={thrTP})");
//                                bestS = newS;
//                            }
//                        }
//                    }
//                }
//            }

//            Print($"=== SETUP SCANNING RESULTS ===");
//            Print($"Setups checked: {setupsChecked}, Valid: {validSetups}");
//            Print($"Best LONG: EV={bestL.EV:F4} (E={bestL.entryDist}, SL={bestL.SlDist}, TP={bestL.TpDist})");
//            Print($"Best SHORT: EV={bestS.EV:F4} (E={bestS.entryDist}, SL={bestS.SlDist}, TP={bestS.TpDist})");

//            return (bestL, bestS);
//        }

//        private bool TryExtractPipsFromKey(string key, out double pips)
//        {
//            // Parsuj klucze typu "≥1.4pips" → 1.4
//            pips = 0;
//            if (string.IsNullOrEmpty(key) || !key.StartsWith("≥") || !key.EndsWith("pips"))
//            {
//                Print($"DEBUG: Key format invalid: '{key}' (empty={string.IsNullOrEmpty(key)}, starts≥={key?.StartsWith("≥")}, endspips={key?.EndsWith("pips")})");
//                return false;
//            }

//            string pipsStr = key.Substring(1, key.Length - 5); // Usuń "≥" i "pips"
//            Print($"DEBUG: Extracting from '{key}' → pipsStr='{pipsStr}'");

//            // Użyj InvariantCulture dla parsowania liczb z kropką
//            bool success = double.TryParse(pipsStr, System.Globalization.NumberStyles.Float,
//                                         System.Globalization.CultureInfo.InvariantCulture, out pips);

//            Print($"DEBUG: Parse result: success={success}, pips={pips}");
//            return success;
//        }

//        private Setup EvalBranch(double pEntry, double pProfit, double pLoss, int distEntry, int distSL, int distTP, double fee, Setup setup)
//        {
//            double pLossGivenEntry = (pLoss) / pEntry;
//            double pNoLossGivenEntry = (pEntry - pLoss) / pEntry;

//            //double pProfitGivenEntry = pProfit /  (pEntry/ (1- pNoLossGivenEntry));
//            double pProfitGivenEntry = pProfit / pEntry;

//            var ev = pProfitGivenEntry * (distEntry + distTP) - pLossGivenEntry * (distSL - distEntry);

//            if (setup.EV < ev)
//            {
//                return new Setup(distTP, distSL, distEntry, ev);
//            }
//            return setup;
//        }

//        /*══════════════════ 5. Wystaw LIMIT ═════════════════════════════*/
//        private void SubmitLimit(int idx, bool longSide, Setup s)
//        {
//            string labelPref = longSide ? "EV_BUY" : "EV_SELL";
//            foreach (var po in PendingOrders) if (po.Label.StartsWith(labelPref)) CancelPendingOrder(po);
//            double pip = Symbol.PipSize; double close = Bars.ClosePrices[idx];
//            double entryPrice = longSide ? close - s.entryDist * pip : close + s.entryDist * pip;
//            int tpPips = s.TpDist + s.entryDist;
//            int slPips = s.SlDist - s.entryDist;
//            double vol = Symbol.NormalizeVolumeInUnits((long)VolumeUnits, RoundingMode.Down);

//            if (slPips < 1)
//            {
//                slPips = 1;
//            }
//            var tr = PlaceLimitOrder(longSide ? TradeType.Buy : TradeType.Sell, SymbolName, vol, entryPrice, labelPref, slPips, tpPips);
//            if (tr.PendingOrder != null && tr.PendingOrder?.StopLoss == null || tr.Position != null && tr.Position?.StopLoss == null)
//            {
//                Debugger.Break();
//            }
//            if (tr.PendingOrder != null && tr.PendingOrder?.TakeProfit == null || tr.Position != null && tr.Position?.TakeProfit == null)
//            {
//                Debugger.Break();
//            }
//        }

//        private void UpdateAllMissingSLTP()
//        {
//            // dla każdej otwartej pozycji
//            foreach (var position in Positions)
//            {
//                // jeśli brakuje SL lub TP
//                if (!position.StopLoss.HasValue || !position.TakeProfit.HasValue)
//                {
//                    // szukamy historycznego ordera otwierającego tę pozycję
//                    var openOrder = HistoricalOrders
//                        .FirstOrDefault(o => o.PositionId == position.Id && o.Status == HistoricalOrderStatus.Filled);

//                    if (openOrder == null)
//                    {
//                        Print("Nie znaleziono historycznego ordera dla pozycji {0}", position.Id);
//                        continue;
//                    }

//                    // kopiujemy i ustawiamy SL
//                    if (!position.StopLoss.HasValue && openOrder.StopLoss.HasValue)
//                    {

//                        TryUpdateSL(position, openOrder.StopLoss.Value);
//                    }

//                    // kopiujemy i ustawiamy TP
//                    if (!position.TakeProfit.HasValue && openOrder.TakeProfit.HasValue)
//                        TryUpdateTP(position, openOrder.TakeProfit.Value);
//                }
//            }
//        }

//        private void TryUpdateSL(Position position, double targetSL)
//        {
//            double sl = targetSL;
//            int attempts = 0;
//            bool isLong = position.TradeType == TradeType.Buy;


//            if (Bars.LastBar.Close > targetSL && !isLong)
//            {
//                position.Close();
//            }
//            if (Bars.LastBar.Close < targetSL && isLong)
//            {
//                position.Close();
//            }


//            // w zależności od kierunku pozycji, SL musi być poniżej (long) lub powyżej (short) ceny entry
//            double priceStep = Symbol.PipSize; // lub TickSize

//            while (attempts < 20) // ograniczamy liczbę prób
//            {
//                var result = position.ModifyStopLossPrice(sl);

//                if (result.IsSuccessful)
//                {
//                    Print("Ustawiono SL dla {0} na {1}", position.Label, sl);
//                    return;
//                }

//                // jeśli nie udało się – przesuwamy SL o jeden krok
//                sl += isLong ? -priceStep : +priceStep;
//                attempts++;
//            }

//            Print("Nie udało się ustawić SL dla {0} po {1} próbach", position.Label, attempts);
//        }

//        private void TryUpdateTP(Position position, double targetTP)
//        {
//            double tp = targetTP;
//            int attempts = 0;

//            bool isLong = position.TradeType == TradeType.Buy;
//            double priceStep = Symbol.PipSize;

//            if (Bars.LastBar.Close < targetTP && !isLong)
//            {
//                position.Close();
//            }
//            if (Bars.LastBar.Close > targetTP && isLong)
//            {
//                position.Close();
//            }

//            while (attempts < 20)
//            {
//                var result = position.ModifyTakeProfitPrice(tp);

//                if (result.IsSuccessful)
//                {
//                    Print("Ustawiono TP dla {0} na {1}", position.Label, tp);
//                    return;
//                }

//                // jeśli nie udało się – przesuwamy TP o jeden krok
//                tp += isLong ? +priceStep : -priceStep;
//                attempts++;
//            }

//            Print("Nie udało się ustawić TP dla {0} po {1} próbach", position.Label, attempts);
//        }
//    }
//}
