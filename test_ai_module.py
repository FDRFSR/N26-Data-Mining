#!/usr/bin/env python3
"""
Test per il modulo AI N26 Advanced Analytics
"""

import ai_predictor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def main():
    print('🤖 Testing N26 AI Predictor Module...')
    print('=' * 50)

    try:
        # Creiamo dati di test realistici
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
        np.random.seed(42)  # Per risultati riproducibili

        # Simuliamo pattern realistici di spesa
        amounts = []
        categories = []
        for i, date in enumerate(dates):
            # Pattern settimanale: più spese nei weekend
            if date.weekday() >= 5:  # Weekend
                base_amount = -75  # Spese più alte
            else:
                base_amount = -45  # Spese normali giorni lavorativi
            
            # Aggiunta variabilità
            amount = np.random.normal(base_amount, 25)
            amounts.append(amount)
            
            # Categorie realistiche
            if date.weekday() >= 5:
                cat = np.random.choice(['Entertainment', 'Shopping', 'Food'], p=[0.4, 0.4, 0.2])
            else:
                cat = np.random.choice(['Food', 'Transport', 'Shopping'], p=[0.5, 0.3, 0.2])
            categories.append(cat)

        test_data = pd.DataFrame({
            'Date': dates,
            'Amount': amounts,
            'Category': categories
        })

        print(f'📊 Generated {len(test_data)} realistic test transactions')
        print(f'Total spending: €{abs(test_data["Amount"].sum()):.2f}')

        # Inizializziamo il predictor
        print('\n🏗️ Initializing AI Predictor...')
        predictor = ai_predictor.N26AIPredictor()
        print('✅ Predictor initialized')

        # Test 1: Training del modello
        print('\n🔄 Training AI model...')
        success = predictor.train_model(test_data)
        print(f'Training result: {"✅ Success" if success else "❌ Failed"}')

        if success:
            # Test 2: Predizione spese prossima settimana
            print('\n🔮 Predicting next week spending...')
            try:
                predictions = predictor.predict_next_week_spending()
                if predictions:
                    total_predicted = sum(p['importo_predetto'] for p in predictions)
                    print(f'✅ Predicted spending for next week: €{total_predicted:.2f}')
                    print(f'📅 Daily breakdown: {len(predictions)} predictions')
                    
                    # Mostra primi 3 giorni
                    for i, pred in enumerate(predictions[:3]):
                        print(f'  Day {i+1}: {pred["giorno"]} - €{pred["importo_predetto"]:.2f}')
                else:
                    print('❌ No predictions generated')
            except Exception as e:
                print(f'❌ Prediction error: {e}')

            # Test 3: Rilevamento anomalie
            print('\n🚨 Detecting spending anomalies...')
            try:
                anomalies = predictor.detect_spending_anomalies()
                print(f'✅ Found {len(anomalies)} anomalous transactions')
                if len(anomalies) > 0:
                    print(f'🔍 Most anomalous: €{anomalies[0]["importo"]:.2f} - {anomalies[0]["severity"]} severity')
                    print(f'   Date: {anomalies[0]["data"]}, Category: {anomalies[0]["categoria"]}')
            except Exception as e:
                print(f'❌ Anomaly detection error: {e}')

            # Test 4: Generazione insights
            print('\n💡 Generating spending insights...')
            try:
                insights = predictor.get_spending_insights()
                print(f'✅ Generated {len(insights)} insights:')
                for i, insight in enumerate(insights[:3], 1):
                    print(f'{i}. {insight["titolo"]}: {insight["descrizione"]}')
            except Exception as e:
                print(f'❌ Insights generation error: {e}')

        print('\n🎉 AI Module test completed!')
        
    except Exception as e:
        print(f'❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
