from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd 
import pickle
from pathlib import Path

# Import z naszego dual-output modelu
from CumulativeForexTransformer8_fixed import (
    DualOutputDecoderTransformer,
    build_decoder_model,
    ohlcv_to_percent_features,
    SEQ_LEN,
    PCT_THRESHOLDS,
)

# ── Ustawienia ────────────────────────────────────────
MODEL_PATH = Path("./best_decoder_model_fixed.pth")
CALIBRATOR_PATH = Path("saved_models/calibrators_dual.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wymagane dane dla dual-output modelu
MIN_HISTORY_FOR_PROCESSING = 10  # Minimalna historia dla przetwarzania
TOTAL_REQUIRED_H1_CANDLES = SEQ_LEN + MIN_HISTORY_FOR_PROCESSING  # SEQ_LEN = 200
NUM_FEATURES = 5  # OHLCV

# ── Inicjalizacja ─────────────────────────────────────
app = Flask(__name__)
print(f"🔧 Uruchamianie Dual-Output Decoder Server na {DEVICE}...")

# Model - dual-output decoder transformer
model = build_decoder_model().to(DEVICE)

# Załaduj model
if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Dual-Output Model załadowany z epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
        print(f"   Diff loss: {checkpoint.get('val_diff_loss', 'unknown'):.4f}")
        print(f"   Threshold loss: {checkpoint.get('val_threshold_loss', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Dual-Output Model załadowany")
else:
    print("⚠️  Model nie został znaleziony, używam niezainicjalizowanego modelu")

model.eval()

# Kalibratory (opcjonalnie)
calibrators = None
if CALIBRATOR_PATH.exists():
    try:
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrators = pickle.load(f)
        print(f"✅ Kalibratory załadowane: {len(calibrators)} szt.")
    except Exception as e:
        print(f"⚠️ Nie udało się załadować kalibratorów: {e}")


def prepare_sequence_for_prediction(h1_data):
    """
    Przygotuj sekwencję H1 dla predykcji dual-output modelu.
    
    Args:
        h1_data: numpy array shape (N, 5) - OHLCV data
        
    Returns:
        torch.Tensor shape (1, SEQ_LEN, 7) - features ready for model
    """
    if len(h1_data) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} candles, got {len(h1_data)}")
    
    # Użyj ostatnich SEQ_LEN świec
    last_sequence = h1_data[-SEQ_LEN:]
    
    # Konwertuj na features (7-dim: Δ% OHLC + logVol + range% + std%)
    features = ohlcv_to_percent_features(last_sequence)
    
    # Dodaj batch dimension
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    return features_tensor


def predict_with_dual_model(model, X, calibrators=None):
    """
    Predykcja z dual-output modelem.
    
    Args:
        model: DualOutputDecoderTransformer
        X: torch.Tensor shape (1, SEQ_LEN, 7)
        calibrators: optional calibrators
        
    Returns:
        dict with 'next_diff', 'thresholds', and processed results
    """
    model.eval()
    with torch.no_grad():
        X = X.to(DEVICE)
        
        # Model zwraca dict: {'next_diff': [1, SEQ_LEN, 7], 'thresholds': [1, SEQ_LEN, 20]}
        predictions = model(X)
        
        # Wyciągnij predykcje z ostatniej pozycji (najważniejszej)
        next_diff = predictions['next_diff'][0, -1, :].cpu().numpy()  # [7] - ostatnia pozycja
        threshold_logits = predictions['thresholds'][0, -1, :].cpu().numpy()  # [20] - ostatnia pozycja
        
        # Konwertuj threshold logits na prawdopodobieństwa
        threshold_probs = torch.sigmoid(torch.tensor(threshold_logits)).numpy()
        
        # Opcjonalna kalibracja
        if calibrators is not None and len(calibrators) >= 20:
            calibrated_probs = np.zeros_like(threshold_probs)
            for i in range(min(20, len(calibrators))):
                calibrated_probs[i] = calibrators[i].predict(threshold_probs[i:i+1])[0]
            threshold_probs = calibrated_probs
        
        return {
            'next_diff': next_diff,
            'threshold_probs': threshold_probs,
            'raw_predictions': {
                'next_diff_sequence': predictions['next_diff'][0].cpu().numpy(),  # [SEQ_LEN, 7]
                'threshold_sequence': torch.sigmoid(predictions['thresholds'][0]).cpu().numpy()  # [SEQ_LEN, 20]
            }
        }


def format_threshold_results(threshold_probs):
    """
    Format threshold probabilities into UP/DOWN structure.
    
    Args:
        threshold_probs: numpy array [20] - probabilities for each threshold
        
    Returns:
        dict with 'up' and 'down' threshold probabilities
    """
    # PCT_THRESHOLDS ma 10 progów, więc mamy 20 outputów: 10 UP + 10 DOWN
    up_probs = threshold_probs[:10]    # Pierwsze 10 to UP
    down_probs = threshold_probs[10:]  # Następne 10 to DOWN
    
    # Konwertuj progi z log-scale na pips (przybliżenie)
    threshold_pips = []
    for i, log_thresh in enumerate(PCT_THRESHOLDS):
        # log_thresh to log(1 + pct), więc pct = exp(log_thresh) - 1
        pct = np.exp(log_thresh) - 1
        pips = pct * 10000  # Przybliżenie: 1% = 100 pips
        threshold_pips.append(pips)
    
    up_dict = {}
    down_dict = {}
    
    for i in range(10):
        pips = threshold_pips[i]
        up_dict[f"≥{pips:.1f}pips"] = float(up_probs[i])
        down_dict[f"≥{pips:.1f}pips"] = float(down_probs[i])
    
    return {"up": up_dict, "down": down_dict}


def format_next_diff_results(next_diff):
    """
    Format next_diff predictions into readable structure.
    
    Args:
        next_diff: numpy array [7] - predicted differences for next step
        
    Returns:
        dict with formatted predictions
    """
    feature_names = ['open_diff', 'high_diff', 'low_diff', 'close_diff', 
                    'volume_diff', 'range_diff', 'std_diff']
    
    return {
        'predicted_changes': {name: float(diff) for name, diff in zip(feature_names, next_diff)},
        'summary': {
            'predicted_close_change_pct': float(next_diff[3] * 100),  # Close diff w %
            'predicted_volatility': float(abs(next_diff[1] - next_diff[2]) * 100),  # High-Low diff w %
            'predicted_volume_change': float(next_diff[4]),
            'overall_direction': 'bullish' if next_diff[3] > 0 else 'bearish'
        }
    }


# ── Endpoint ──────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # KOMPATYBILNOŚĆ WSTECZNA: akceptuj zarówno stary format jak i nowy
        required_fields = ["h1_data", "h1_timestamps"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: '{field}'"}), 400

        # Parsowanie danych H1 (wymagane)
        h1_data = np.array(data["h1_data"], dtype=np.float32)
        h1_timestamps = pd.to_datetime(data["h1_timestamps"])
        
        # OPCJONALNE: parsowanie danych M5 dla kompatybilności (ale ich nie używamy)
        m5_provided = False
        if "m5_data" in data and "m5_timestamps" in data:
            m5_provided = True
            print(f"ℹ️  M5 data provided but ignored (dual-output model uses H1-only)")
        
        # Walidacja kształtów
        if h1_data.ndim != 2 or h1_data.shape[1] != NUM_FEATURES:
            return jsonify({"error": f"H1 data expected shape (N, {NUM_FEATURES}), got {h1_data.shape}"}), 400
        
        # Sprawdź czy mamy wystarczająco danych
        if len(h1_data) < TOTAL_REQUIRED_H1_CANDLES:
            return jsonify({
                "error": f"Need at least {TOTAL_REQUIRED_H1_CANDLES} H1 candles, got {len(h1_data)}"
            }), 400
        
        # Sprawdź chronologię timestamps
        if not h1_timestamps.is_monotonic_increasing:
            return jsonify({"error": "H1 timestamps must be in chronological order"}), 400
        
        # Przygotuj sekwencję dla dual-output modelu
        X_pred = prepare_sequence_for_prediction(h1_data)
        
        # Predykcja
        results = predict_with_dual_model(model, X_pred, calibrators)
        
        # Format wyników
        threshold_results = format_threshold_results(results['threshold_probs'])
        next_diff_results = format_next_diff_results(results['next_diff'])
        
        # WSTECZNA KOMPATYBILNOŚĆ: zachowaj stary format + dodaj nowe informacje
        response = {
            # Stary format (kompatybilność)
            "up": threshold_results["up"],
            "down": threshold_results["down"],
            
            # NOWE: Dodatkowe informacje z dual-output modelu
            "next_step_prediction": next_diff_results,
            "advanced_analytics": {
                "sequence_predictions": {
                    "threshold_evolution": results['raw_predictions']['threshold_sequence'][-10:].tolist(),  # Ostatnie 10 pozycji
                    "diff_evolution": results['raw_predictions']['next_diff_sequence'][-10:].tolist()  # Ostatnie 10 pozycji
                },
                "confidence_metrics": {
                    "avg_threshold_confidence": float(np.mean(results['threshold_probs'])),
                    "max_threshold_confidence": float(np.max(results['threshold_probs'])),
                    "prediction_uncertainty": float(np.std(results['next_diff']))
                }
            },
            
            "metadata": {
                "h1_candles_used": len(h1_data),
                "latest_h1_time": str(h1_timestamps[-1]),
                "model_type": "dual_output_decoder_transformer",
                "model_version": "v8_fixed",
                "calibrated": calibrators is not None,
                "seq_len": SEQ_LEN,
                "input_shape": list(X_pred.shape),
                "backward_compatible": True,
                "m5_data_provided": m5_provided,
                "m5_data_ignored": m5_provided,
                
                # NOWE: Dodatkowe metadane
                "dual_outputs": True,
                "autoregressive_prediction": True,
                "causal_attention": True,
                "threshold_count": 20,
                "feature_dimensions": 7,
                "architecture": {
                    "type": "decoder_transformer",
                    "d_model": 256,
                    "depth": 8,
                    "heads": 8,
                    "causal_mask": True
                }
            }
        }
        
        # Logging
        print(f"Dual Prediction:")
        print(f"  Thresholds - up_≥1pip={threshold_results['up'][list(threshold_results['up'].keys())[0]]:.3f}")
        print(f"  Next step - close_change={next_diff_results['summary']['predicted_close_change_pct']:.3f}%")
        print(f"  Direction: {next_diff_results['summary']['overall_direction']}")
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(f"Error in dual prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── Endpoint info ─────────────────────────────────────
@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model": "DualOutputDecoderTransformer",
        "version": "v8_fixed_dual_output",
        "device": str(DEVICE),
        "required_h1_candles": TOTAL_REQUIRED_H1_CANDLES,
        "seq_len": SEQ_LEN,
        "min_history": MIN_HISTORY_FOR_PROCESSING,
        "calibrated": calibrators is not None,
        "features": "H1 OHLCV → 7-dim features (Δ% OHLC + logVol + range% + std%)",
        "data_processing": "ohlcv_to_percent_features",
        "backward_compatible": True,
        "accepts_m5_data": "yes, but ignores it",
        
        # NOWE: Informacje o dual-output
        "dual_outputs": {
            "next_diff": "Autoregressive prediction of next-step feature changes",
            "thresholds": "Probability of exceeding price thresholds",
            "threshold_levels": len(PCT_THRESHOLDS),
            "threshold_range": f"{PCT_THRESHOLDS[0]:.4f} to {PCT_THRESHOLDS[-1]:.4f} (log scale)"
        },
        
        "model_architecture": {
            "type": "decoder_transformer",
            "input_dim": 7,
            "d_model": 256,
            "depth": 8,
            "heads": 8,
            "causal_attention": True,
            "dual_heads": True,
            "weighted_loss": "position_weighted_0.5_to_1.0"
        },
        
        "enhanced_features": {
            "sequence_to_sequence_learning": True,
            "autoregressive_training": True,
            "position_weighted_loss": True,
            "next_step_prediction": True,
            "threshold_probabilities": True,
            "confidence_metrics": True,
            "sequence_evolution_tracking": True
        }
    })


# ── Health check ──────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "model_type": "DualOutputDecoderTransformer",
        "dual_outputs": True,
        "backward_compatible": True
    })


# ── Test endpoint ─────────────────────────────────────
@app.route("/test", methods=["POST"])
def test():
    """Test endpoint z przykładowymi danymi."""
    try:
        # Generuj przykładowe dane H1
        np.random.seed(42)
        n_candles = TOTAL_REQUIRED_H1_CANDLES + 50
        
        # Symuluj realistyczne dane EUR/USD
        base_price = 1.0500
        prices = [base_price]
        volumes = []
        
        for i in range(n_candles):
            # Random walk z małym trendem
            change = np.random.normal(0, 0.001)  # 0.1% std
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
            # Volume z trendem
            volume = np.random.lognormal(10, 0.5)  # Realistic volume
            volumes.append(volume)
        
        prices = np.array(prices[1:])  # Remove first element
        volumes = np.array(volumes)
        
        # Stwórz OHLCV
        ohlcv_data = []
        for i in range(n_candles):
            open_price = prices[i]
            high_price = open_price * (1 + abs(np.random.normal(0, 0.0005)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.0005)))
            close_price = open_price + np.random.normal(0, 0.0003)
            volume = volumes[i]
            
            ohlcv_data.append([open_price, high_price, low_price, close_price, volume])
        
        # Timestamps
        timestamps = pd.date_range(start='2024-01-01', periods=n_candles, freq='H')
        
        # Przygotuj request
        test_data = {
            "h1_data": ohlcv_data,
            "h1_timestamps": [ts.isoformat() for ts in timestamps]
        }
        
        # Wywołaj predict
        from flask import current_app
        with current_app.test_request_context(json=test_data, content_type='application/json'):
            response = predict()
            return response
            
    except Exception as e:
        import traceback
        print(f"Error in test: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── Start ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Dual-Output Decoder Server gotowy!")
    print(f"📊 Model: DualOutputDecoderTransformer (v8_fixed)")
    print(f"🎯 Endpoint: POST /predict")
    print(f"ℹ️  Info: GET /info")
    print(f"❤️  Health: GET /health")
    print(f"🧪 Test: POST /test")
    print(f"🔧 Features: 7-dim (Δ% OHLC + logVol + range% + std%)")
    print(f"📈 Required H1 candles: {TOTAL_REQUIRED_H1_CANDLES}")
    print(f"📏 Sequence length: {SEQ_LEN}")
    print(f"🔄 BACKWARD COMPATIBLE: maintains old API format")
    print(f"✨ ENHANCED: dual outputs + next-step prediction + confidence metrics")
    print(f"🎭 Dual Outputs: next_diff + threshold_probabilities")
    print(f"🧠 Architecture: Decoder Transformer with causal attention")
    app.run(host="::", port=5051)  # Different port to avoid conflicts
