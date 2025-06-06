#!/bin/bash
# N26 AI-Powered Analytics Dashboard Launcher
# Script di avvio completo per il sistema con AI integrato

echo "🚀 N26 AI-Powered Analytics Dashboard"
echo "====================================="
echo ""

# Verifica virtual environment
if [ ! -d "venv_n26" ]; then
    echo "❌ Virtual environment non trovato"
    echo "Esegui prima: python -m venv venv_n26 && source venv_n26/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Attiva virtual environment
source venv_n26/bin/activate
echo "✅ Virtual environment attivato"

# Verifica dipendenze AI
echo "🔍 Verifica dipendenze AI..."
python -c "
try:
    import scikit_learn
    print('✅ scikit-learn OK')
except ImportError:
    print('❌ scikit-learn non trovato')
    exit(1)

try:
    import joblib
    print('✅ joblib OK')
except ImportError:
    print('❌ joblib non trovato')
    exit(1)
" || {
    echo "📦 Installazione dipendenze AI..."
    pip install scikit-learn joblib
}

# Test moduli
echo ""
echo "🧪 Test moduli del sistema..."
python -c "
try:
    import ai_predictor
    print('✅ ai_predictor.py - OK')
except Exception as e:
    print(f'❌ ai_predictor.py - {e}')

try:
    from advanced_analytics import N26AdvancedAnalytics
    print('✅ advanced_analytics.py - OK')
except Exception as e:
    print(f'❌ advanced_analytics.py - {e}')

try:
    from analytics_dashboard import AdvancedAnalyticsDashboard
    print('✅ analytics_dashboard.py - OK')
except Exception as e:
    print(f'❌ analytics_dashboard.py - {e}')
"

echo ""
echo "🎯 Scelta modalità di avvio:"
echo "1. 🖥️  Dashboard GUI con AI (Consigliato)"
echo "2. 📊 Demo Advanced Analytics"
echo "3. 🤖 Test AI Module"
echo "4. 🔬 Test Integrazione Completa"
echo ""

read -p "Seleziona opzione (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Avvio Dashboard GUI con AI integrato..."
        echo "   • Advanced Analytics Dashboard"
        echo "   • AI-Powered Predictions"
        echo "   • Smart Insights Generator"
        echo "   • Anomaly Detection"
        echo ""
        python analytics_dashboard.py
        ;;
    2)
        echo ""
        echo "📊 Esecuzione demo Advanced Analytics..."
        python demo_analytics.py
        ;;
    3)
        echo ""
        echo "🤖 Test AI Module..."
        python test_ai_module.py
        ;;
    4)
        echo ""
        echo "🔬 Test integrazione completa..."
        python test_ai_integration.py
        ;;
    *)
        echo "❌ Opzione non valida"
        exit 1
        ;;
esac

echo ""
echo "👋 Arrivederci!"
