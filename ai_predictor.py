#!/usr/bin/env python3
"""
N26 AI-Powered Spending Predictions
Utilizza machine learning per predire spese future e identificare pattern
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class N26AIPredictor:
    """Sistema di predizioni AI per spese e pattern finanziari"""
    
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path
        self.df = None
        self.spending_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()  # Per spending predictor
        self.anomaly_scaler = StandardScaler()  # Per anomaly detector
        self.feature_columns = []
        if csv_path:
            self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Carica e prepara i dati per il machine learning"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['Data'] = pd.to_datetime(self.df['Data'])
            self.df['Importo'] = pd.to_numeric(self.df['Importo'], errors='coerce')
            self.df = self.df.dropna()
            
            # Feature engineering
            self._create_features()
            print(f"✅ Dati caricati e processati: {len(self.df)} transazioni")
            
        except Exception as e:
            print(f"❌ Errore caricamento dati: {e}")
            self.df = pd.DataFrame()
    
    def load_from_dataframe(self, df: pd.DataFrame):
        """Carica dati da un DataFrame esistente"""
        try:
            self.df = df.copy()
            
            # Standardizza nomi colonne se necessario
            if 'Date' in self.df.columns:
                self.df = self.df.rename(columns={'Date': 'Data', 'Amount': 'Importo', 'Category': 'Categoria'})
            
            self.df['Data'] = pd.to_datetime(self.df['Data'])
            self.df['Importo'] = pd.to_numeric(self.df['Importo'], errors='coerce')
            self.df = self.df.dropna()
            
            # Feature engineering
            self._create_features()
            print(f"✅ DataFrame caricato e processato: {len(self.df)} transazioni")
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento DataFrame: {e}")
            self.df = pd.DataFrame()
            return False
    
    def _create_features(self):
        """Crea features per il machine learning"""
        # Features temporali
        self.df['Giorno'] = self.df['Data'].dt.day
        self.df['Mese'] = self.df['Data'].dt.month
        self.df['GiornoSettimana'] = self.df['Data'].dt.dayofweek
        self.df['GiornoAnno'] = self.df['Data'].dt.dayofyear
        self.df['Settimana'] = self.df['Data'].dt.isocalendar().week
        
        # Features di spesa
        self.df['ImportoAbs'] = self.df['Importo'].abs()
        self.df['IsSpesa'] = (self.df['Importo'] < 0).astype(int)
        self.df['IsWeekend'] = (self.df['GiornoSettimana'].isin([5, 6])).astype(int)
        
        # Encoding categoria
        self.df['Categoria_encoded'] = pd.Categorical(self.df['Categoria']).codes
        
        # Features rolling (ultimi 7 giorni)
        self.df = self.df.sort_values('Data')
        self.df['SpesaMedia7g'] = self.df['ImportoAbs'].rolling(window=7).mean()
        self.df['SpesaMax7g'] = self.df['ImportoAbs'].rolling(window=7).max()
        
        # Features finali per ML
        self.feature_columns = [
            'Giorno', 'Mese', 'GiornoSettimana', 'GiornoAnno', 'Settimana',
            'Categoria_encoded', 'IsWeekend', 'SpesaMedia7g', 'SpesaMax7g'
        ]
        
        # Rimuovi NaN creati da rolling
        self.df = self.df.dropna()
    
    def train_spending_predictor(self):
        """Addestra il modello di predizione spese"""
        try:
            # Filtra solo le spese (importi negativi)
            spese_df = self.df[self.df['IsSpesa'] == 1].copy()
            
            if len(spese_df) < 10:
                print("❌ Dati insufficienti per training")
                return False
            
            # Prepara features e target
            X = spese_df[self.feature_columns]
            y = spese_df['ImportoAbs']  # Prediciamo l'importo assoluto
            
            print(f"   Debug: Training features shape: {X.shape}")
            print(f"   Debug: Feature columns: {self.feature_columns}")
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scaling
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Training modello
            self.spending_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            )
            self.spending_model.fit(X_train_scaled, y_train)
            
            # Valutazione
            y_pred = self.spending_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"✅ Modello spese addestrato")
            print(f"   📊 MAE: €{mae:.2f}")
            print(f"   📊 R²: {r2:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore training modello: {e}")
            return False
    
    def train_anomaly_detector(self):
        """Addestra il detector di anomalie"""
        try:
            spese_df = self.df[self.df['IsSpesa'] == 1].copy()
            
            if len(spese_df) < 10:
                print("❌ Dati insufficienti per anomaly detection")
                return False
            
            X = spese_df[self.feature_columns + ['ImportoAbs']]
            X_scaled = self.anomaly_scaler.fit_transform(X)
            
            # Training anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # 10% outliers
                random_state=42
            )
            self.anomaly_detector.fit(X_scaled)
            
            print("✅ Anomaly detector addestrato")
            return True
            
        except Exception as e:
            print(f"❌ Errore training anomaly detector: {e}")
            return False
    
    def train_model(self, df: pd.DataFrame = None):
        """Addestra tutti i modelli AI (predizioni + anomaly detection)"""
        try:
            if df is not None:
                success = self.load_from_dataframe(df)
                if not success:
                    return False
            
            if self.df is None or len(self.df) == 0:
                print("❌ Nessun dato disponibile per il training")
                return False
            
            print("🔄 Training spending predictor...")
            success1 = self.train_spending_predictor()
            
            print("🔄 Training anomaly detector...")  
            success2 = self.train_anomaly_detector()
            
            if success1 and success2:
                print("✅ Tutti i modelli addestrati con successo!")
                return True
            else:
                print("⚠️ Alcuni modelli potrebbero non essere stati addestrati correttamente")
                return False
                
        except Exception as e:
            print(f"❌ Errore training modelli: {e}")
            return False
    
    def predict_next_week_spending(self):
        """Predice le spese della prossima settimana"""
        try:
            if self.spending_model is None:
                print("❌ Modello non addestrato")
                return []
            
            predictions = []
            last_date = self.df['Data'].max()
            
            for i in range(1, 8):  # Prossimi 7 giorni
                next_date = last_date + timedelta(days=i)
                
                # Crea features per la predizione (stesso ordine del training)
                features = {
                    'Giorno': next_date.day,
                    'Mese': next_date.month,
                    'GiornoSettimana': next_date.weekday(),
                    'GiornoAnno': next_date.timetuple().tm_yday,
                    'Settimana': next_date.isocalendar()[1],
                    'Categoria_encoded': self.df['Categoria_encoded'].mode().iloc[0] if len(self.df) > 0 else 0,
                    'IsWeekend': 1 if next_date.weekday() >= 5 else 0,
                    'SpesaMedia7g': self.df['ImportoAbs'].tail(7).mean() if len(self.df) > 0 else 0,
                    'SpesaMax7g': self.df['ImportoAbs'].tail(7).max() if len(self.df) > 0 else 0
                }
                
                # Mantieni stesso ordine delle feature columns del training
                feature_values = [features[col] for col in self.feature_columns]
                
                print(f"   Debug: Prediction features length: {len(feature_values)}")
                print(f"   Debug: Expected features: {len(self.feature_columns)}")
                
                # Predizione
                X_pred = np.array([feature_values])
                X_pred_scaled = self.scaler.transform(X_pred)
                predicted_amount = self.spending_model.predict(X_pred_scaled)[0]
                
                predictions.append({
                    'data': next_date.strftime('%Y-%m-%d'),
                    'giorno': next_date.strftime('%A'),
                    'importo_predetto': round(predicted_amount, 2),
                    'is_weekend': features['IsWeekend']
                })
            
            return predictions
            
        except Exception as e:
            print(f"❌ Errore predizione: {e}")
            return []
    
    def detect_spending_anomalies(self, last_n_days=30):
        """Rileva anomalie nelle spese degli ultimi N giorni"""
        try:
            if self.anomaly_detector is None:
                print("❌ Anomaly detector non addestrato")
                return []
            
            # Filtra ultimi N giorni
            cutoff_date = self.df['Data'].max() - timedelta(days=last_n_days)
            recent_df = self.df[self.df['Data'] >= cutoff_date].copy()
            spese_recent = recent_df[recent_df['IsSpesa'] == 1].copy()
            
            if len(spese_recent) == 0:
                return []
            
            # Prepara features
            X = spese_recent[self.feature_columns + ['ImportoAbs']]
            X_scaled = self.anomaly_scaler.transform(X)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
            is_anomaly = self.anomaly_detector.predict(X_scaled) == -1
            
            anomalies = []
            for idx, (_, row) in enumerate(spese_recent.iterrows()):
                if is_anomaly[idx]:
                    anomalies.append({
                        'data': row['Data'].strftime('%Y-%m-%d'),
                        'importo': row['Importo'],
                        'categoria': row['Categoria'],
                        'descrizione': row.get('Descrizione', ''),
                        'anomaly_score': float(anomaly_scores[idx]),
                        'severity': 'Alta' if anomaly_scores[idx] < -0.5 else 'Media'
                    })
            
            return sorted(anomalies, key=lambda x: x['anomaly_score'])
            
        except Exception as e:
            print(f"❌ Errore detection anomalie: {e}")
            return []
    
    def get_spending_insights(self):
        """Genera insights intelligenti sui pattern di spesa"""
        try:
            insights = []
            
            # Pattern temporali
            spese_df = self.df[self.df['IsSpesa'] == 1].copy()
            
            # Insight 1: Giorni più costosi
            spese_per_giorno = spese_df.groupby('GiornoSettimana')['ImportoAbs'].mean()
            giorno_piu_costoso = spese_per_giorno.idxmax()
            giorni = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
            
            insights.append({
                'tipo': 'pattern_temporale',
                'titolo': f'Giorno più costoso: {giorni[giorno_piu_costoso]}',
                'descrizione': f'In media spendi €{spese_per_giorno[giorno_piu_costoso]:.2f} di {giorni[giorno_piu_costoso]}',
                'importanza': 'media'
            })
            
            # Insight 2: Categorie problematiche
            spese_per_categoria = spese_df.groupby('Categoria')['ImportoAbs'].agg(['sum', 'mean', 'count'])
            categoria_top = spese_per_categoria['sum'].idxmax()
            
            insights.append({
                'tipo': 'categoria_analisi',
                'titolo': f'Categoria principale: {categoria_top}',
                'descrizione': f'€{spese_per_categoria.loc[categoria_top, "sum"]:.2f} totali in {spese_per_categoria.loc[categoria_top, "count"]} transazioni',
                'importanza': 'alta'
            })
            
            # Insight 3: Trend mensile
            spese_df['Mese_Anno'] = spese_df['Data'].dt.to_period('M')
            trend_mensile = spese_df.groupby('Mese_Anno')['ImportoAbs'].sum()
            
            if len(trend_mensile) >= 2:
                variazione = ((trend_mensile.iloc[-1] - trend_mensile.iloc[-2]) / trend_mensile.iloc[-2]) * 100
                trend_text = "aumentato" if variazione > 0 else "diminuito"
                
                insights.append({
                    'tipo': 'trend_analisi',
                    'titolo': f'Spese mensili {trend_text}',
                    'descrizione': f'Le tue spese sono {trend_text}e del {abs(variazione):.1f}% rispetto al mese scorso',
                    'importanza': 'alta' if abs(variazione) > 20 else 'media'
                })
            
            return insights
            
        except Exception as e:
            print(f"❌ Errore generazione insights: {e}")
            return []
    
    def save_models(self, model_path="ai_models"):
        """Salva i modelli addestrati"""
        try:
            import os
            os.makedirs(model_path, exist_ok=True)
            
            if self.spending_model:
                joblib.dump(self.spending_model, f"{model_path}/spending_model.pkl")
                joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
                
                # Salva metadati
                metadata = {
                    'feature_columns': self.feature_columns,
                    'model_type': 'RandomForestRegressor',
                    'trained_date': datetime.now().isoformat()
                }
                
                with open(f"{model_path}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"✅ Modelli salvati in {model_path}/")
                return True
                
        except Exception as e:
            print(f"❌ Errore salvataggio modelli: {e}")
            return False
    
    def load_models(self, model_path="ai_models"):
        """Carica i modelli salvati"""
        try:
            self.spending_model = joblib.load(f"{model_path}/spending_model.pkl")
            self.scaler = joblib.load(f"{model_path}/scaler.pkl")
            
            with open(f"{model_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata['feature_columns']
            
            print(f"✅ Modelli caricati da {model_path}/")
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento modelli: {e}")
            return False

# Funzione di utilità per demo
def run_ai_predictions_demo(csv_path):
    """Esegue una demo completa delle predizioni AI"""
    print("🤖 N26 AI-Powered Spending Predictions - DEMO")
    print("=" * 50)
    
    # Inizializza AI predictor
    ai_predictor = N26AIPredictor(csv_path)
    
    if len(ai_predictor.df) == 0:
        print("❌ Nessun dato disponibile per la demo")
        return
    
    # Training modelli
    print("\n🔧 Training modelli AI...")
    ai_predictor.train_spending_predictor()
    ai_predictor.train_anomaly_detector()
    
    # Predizioni prossima settimana
    print("\n📈 Predizioni spese prossima settimana:")
    predictions = ai_predictor.predict_next_week_spending()
    
    total_predicted = 0
    for pred in predictions:
        emoji = "🌅" if pred['is_weekend'] else "🏢"
        print(f"   {emoji} {pred['giorno']} ({pred['data']}): €{pred['importo_predetto']:.2f}")
        total_predicted += pred['importo_predetto']
    
    print(f"\n💰 Totale predetto settimana: €{total_predicted:.2f}")
    
    # Anomalie
    print("\n🚨 Anomalie rilevate (ultimi 30 giorni):")
    anomalies = ai_predictor.detect_spending_anomalies()
    
    if anomalies:
        for anomaly in anomalies[:5]:  # Top 5 anomalie
            print(f"   ⚠️  {anomaly['data']}: €{anomaly['importo']:.2f} ({anomaly['categoria']}) - {anomaly['severity']}")
    else:
        print("   ✅ Nessuna anomalia significativa rilevata")
    
    # Insights
    print("\n💡 Insights AI:")
    insights = ai_predictor.get_spending_insights()
    
    for insight in insights:
        emoji = "🔥" if insight['importanza'] == 'alta' else "💡"
        print(f"   {emoji} {insight['titolo']}")
        print(f"      {insight['descrizione']}")
    
    # Salva modelli
    ai_predictor.save_models()
    
    print("\n🎯 Demo AI completata!")
    return ai_predictor

if __name__ == "__main__":
    # Test con dati demo
    demo_path = "demo_completo.csv"
    if os.path.exists(demo_path):
        run_ai_predictions_demo(demo_path)
    else:
        print("❌ File demo non trovato. Esegui prima run_complete_demo.py")
