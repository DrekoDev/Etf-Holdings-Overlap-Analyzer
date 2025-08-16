import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import json
import os

#st.set_page_config(
#    page_title="Analyseur d'Overlap ETF",
#    page_icon="📊",
 #   layout="wide"
#)

#st.title("📊 Analyseur d'Overlap ETF")
#st.markdown("**Analysez les chevauchements dans votre portefeuille d'ETFs**")

CSV_FILE_PATH = "holdings_xd_processed.csv"

def get_available_etfs():
    """Récupère la liste des ETFs disponibles sans charger toutes les données"""
    try:
        # Lire seulement les colonnes ETF_Symbol et ETF_Name
        df = pd.read_csv(CSV_FILE_PATH, delimiter=';', usecols=['ETF_Symbol', 'ETF_Name'])
        df = df.dropna(subset=['ETF_Symbol'])
        df['ETF_Symbol'] = df['ETF_Symbol'].str.strip()
        
        # Obtenir les ETFs uniques avec leurs noms
        etf_info = df.drop_duplicates(subset=['ETF_Symbol']).set_index('ETF_Symbol')['ETF_Name'].to_dict()
        return etf_info
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return {}

def load_etf_holdings(etf_symbols):
    """Charge seulement les holdings des ETFs sélectionnés"""
    try:
        df = pd.read_csv(CSV_FILE_PATH, delimiter=';')
        
        # Filtrer seulement les ETFs sélectionnés
        df = df[df['ETF_Symbol'].isin(etf_symbols)]
        
        # Nettoyer les données
        df = df.dropna(subset=['ETF_Symbol', 'Ticker', 'Weight_Percent'])
        df = df[df['Weight_Percent'] > 0]
        
        # Nettoyer les colonnes
        df['ETF_Symbol'] = df['ETF_Symbol'].str.strip()
        df['Ticker'] = df['Ticker'].str.strip()
        df['Weight_Percent'] = pd.to_numeric(df['Weight_Percent'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des holdings : {e}")
        return None

def calculate_pairwise_similarity(portfolio_weights, etf_holdings):
    """Calcule la similarité entre chaque paire d'ETFs"""
    etf_list = list(portfolio_weights.keys())
    similarities = {}
    
    for i in range(len(etf_list)):
        for j in range(i+1, len(etf_list)):
            etf_a, etf_b = etf_list[i], etf_list[j]
            
            # Trouver les titres communs
            common_tickers = set(etf_holdings[etf_a].keys()) & set(etf_holdings[etf_b].keys())
            
            similarity = 0
            for ticker in common_tickers:
                # Pour la similarité par paire, on utilise les poids dans les ETFs directement
                weight_a = etf_holdings[etf_a][ticker] / 100
                weight_b = etf_holdings[etf_b][ticker] / 100
                similarity += min(weight_a, weight_b)
            
            similarities[(etf_a, etf_b)] = similarity * 100
    
    return similarities

def calculate_overlap(portfolio_weights, holdings_data):
    """Calcule l'overlap entre les ETFs du portefeuille"""
    overlaps = defaultdict(float)
    ticker_details = {}
    etf_holdings = {}  # Pour stocker les holdings par ETF
    
    # Organiser les données par ETF avec agrégation des doublons
    for etf_symbol, etf_weight in portfolio_weights.items():
        etf_data = holdings_data[holdings_data['ETF_Symbol'] == etf_symbol]
        etf_holdings[etf_symbol] = {}
        
        # Grouper par ticker et sommer les poids pour éliminer les doublons
        for ticker in etf_data['Ticker'].unique():
            ticker_rows = etf_data[etf_data['Ticker'] == ticker]
            total_weight = ticker_rows['Weight_Percent'].sum()  # SOMMER les poids
            etf_holdings[etf_symbol][ticker] = total_weight
    
# Pour chaque ticker, calculer son poids total dans le portefeuille
    for etf_symbol, etf_weight in portfolio_weights.items():
        # Utiliser directement etf_holdings qui a déjà les données agrégées
        for ticker, holding_weight in etf_holdings[etf_symbol].items():
            # Poids du ticker dans le portefeuille total = poids dans l'ETF × poids de l'ETF dans le portefeuille
            portfolio_ticker_weight = (holding_weight / 100) * (etf_weight / 100)
            overlaps[ticker] += portfolio_ticker_weight
            
            # Stocker les détails pour l'affichage
            if ticker not in ticker_details:
                # Récupérer le nom de la company depuis les données originales
                ticker_row = holdings_data[(holdings_data['ETF_Symbol'] == etf_symbol) & 
                                        (holdings_data['Ticker'] == ticker)].iloc[0]
                ticker_details[ticker] = {
                    'company_name': ticker_row.get('Company_Name', ticker) or ticker,
                    'etfs': [],
                    'total_weight': 0
                }
            
            ticker_details[ticker]['etfs'].append({
                'etf': etf_symbol,
                'weight_in_etf': holding_weight,  # Maintenant c'est le poids agrégé
                'weight_in_portfolio': portfolio_ticker_weight * 100
            })
            ticker_details[ticker]['total_weight'] = overlaps[ticker] * 100
    
    return overlaps, ticker_details, etf_holdings

def calculate_relative_overlap(portfolio_weights, pairwise_similarities):
    """Calcule l'overlap relatif basé sur la similarité moyenne pondérée"""
    
    etf_list = list(portfolio_weights.keys())
    total_weighted_similarity = 0.0
    total_weight = 0.0
    
    # Pour chaque paire d'ETFs
    for i in range(len(etf_list)):
        for j in range(i + 1, len(etf_list)):
            etf_a, etf_b = etf_list[i], etf_list[j]
            
            # Poids de cette paire = produit des allocations
            pair_weight = (portfolio_weights[etf_a] / 100) * (portfolio_weights[etf_b] / 100)
            
            # Récupérer la similarité
            if (etf_a, etf_b) in pairwise_similarities:
                similarity = pairwise_similarities[(etf_a, etf_b)]
            elif (etf_b, etf_a) in pairwise_similarities:
                similarity = pairwise_similarities[(etf_b, etf_a)]
            else:
                similarity = 0.0
            
            total_weighted_similarity += similarity * pair_weight
            total_weight += pair_weight
    
    return total_weighted_similarity / total_weight if total_weight > 0 else 0.0

def calculate_portfolio_stats(portfolio_weights, holdings_data):
    """Calcule les statistiques globales du portefeuille"""
    stats = {}
    
    # Nombre total de holdings par ETF
    holdings_per_etf = {}
    for etf_symbol in portfolio_weights.keys():
        etf_data = holdings_data[holdings_data['ETF_Symbol'] == etf_symbol]
        # Compter les tickers uniques pour cet ETF
        unique_holdings = etf_data['Ticker'].nunique()
        holdings_per_etf[etf_symbol] = unique_holdings
    
    # Nombre total de holdings du portefeuille (somme)
    total_holdings = sum(holdings_per_etf.values())
    
    # Nombre de holdings uniques (sans doublons)
    all_tickers = set()
    for etf_symbol in portfolio_weights.keys():
        etf_data = holdings_data[holdings_data['ETF_Symbol'] == etf_symbol]
        all_tickers.update(etf_data['Ticker'].unique())
    unique_holdings_portfolio = len(all_tickers)
    
    # NOUVEAU : Statistiques pays et secteurs
    # Calculer les poids pondérés par pays et secteur
    country_weights = {}
    sector_weights = {}
    
    for etf_symbol, etf_weight in portfolio_weights.items():
        etf_data = holdings_data[holdings_data['ETF_Symbol'] == etf_symbol]
        
        # Grouper par ticker pour éviter les doublons, puis sommer par pays/secteur
        for ticker in etf_data['Ticker'].unique():
            ticker_rows = etf_data[etf_data['Ticker'] == ticker]
            ticker_weight_in_etf = ticker_rows['Weight_Percent'].sum()  # Poids total du ticker dans l'ETF
            ticker_weight_in_portfolio = (ticker_weight_in_etf / 100) * (etf_weight / 100)
            
            # Pays (prendre le premier car un ticker = un pays)
            country = ticker_rows['Country'].iloc[0] if not ticker_rows['Country'].isna().all() else 'Unknown'
            if country not in country_weights:
                country_weights[country] = 0
            country_weights[country] += ticker_weight_in_portfolio
            
            # Secteur (prendre le premier car un ticker = un secteur)
            sector = ticker_rows['Industry_Display'].iloc[0] if not ticker_rows['Industry_Display'].isna().all() else 'Unknown'
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += ticker_weight_in_portfolio
    
    # Convertir en pourcentages
    country_weights = {k: v * 100 for k, v in country_weights.items()}
    sector_weights = {k: v * 100 for k, v in sector_weights.items()}
    
    stats['holdings_per_etf'] = holdings_per_etf
    stats['total_holdings'] = total_holdings
    stats['unique_holdings'] = unique_holdings_portfolio
    stats['country_weights'] = country_weights
    stats['sector_weights'] = sector_weights
    
    return stats

def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff6b35;
        --danger-color: #d62728;
        --background-light: #f8f9fa;
        --text-dark: #2c3e50;
        --border-radius: 12px;
        --shadow: 0 2px 12px rgba(0,0,0,0.1);
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }
    
    .custom-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .custom-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card containers */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
    
    .card-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        color: var(--text-dark);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: var(--shadow);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .status-excellent { background: #d4edda; color: #155724; }
    .status-good { background: #d1ecf1; color: #0c5460; }
    .status-moderate { background: #fff3cd; color: #856404; }
    .status-high { background: #f8d7da; color: #721c24; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: var(--border-radius);
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    /* Progress bars */
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)

def render_custom_header():
    st.markdown("""
    <div class="custom-header">
        <h1>📊 Analyseur d'Overlap ETF</h1>
        <p>Analysez les chevauchements dans votre portefeuille d'ETFs</p>
    </div>
    """, unsafe_allow_html=True)

def render_portfolio_overview(portfolio_stats, portfolio_weights):
    """Affiche la section Vue d'ensemble du portefeuille"""
    
    # Créer 3 colonnes pour les métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Holdings totaux**")
        st.markdown(f"# {portfolio_stats['total_holdings']}")
    
    with col2:
        st.markdown("**🎯 Holdings uniques**")
        st.markdown(f"# {portfolio_stats['unique_holdings']}")
    
    with col3:
        st.markdown("**📈 ETFs sélectionnés**")
        st.markdown(f"# {len(portfolio_weights)}")
    
    st.markdown("---")  # Séparateur
    
    # Détail par ETF
    st.markdown("##### Détail par ETF")
    etf_detail_data = []
    for etf, count in portfolio_stats['holdings_per_etf'].items():
        etf_detail_data.append({
            'ETF': etf,
            'Nombre de holdings': count,
            'Allocation (%)': portfolio_weights[etf]
        })
    
    etf_detail_df = pd.DataFrame(etf_detail_data)
    st.dataframe(etf_detail_df, use_container_width=True, hide_index=True)

def render_geographic_and_sector_analysis(portfolio_stats):
    """Affiche l'analyse géographique et sectorielle"""
    st.subheader("🌍 Répartition Géographique et Sectorielle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        # Préparer les données pays (top 10)
        country_data = portfolio_stats['country_weights']
        # Filtrer les valeurs Unknown et très petites
        filtered_countries = {k: v for k, v in country_data.items() if k != 'Unknown' and v > 0.1}
        top_countries = dict(sorted(filtered_countries.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if top_countries:
            # Extraire les données dans l'ordre
            countries = list(top_countries.keys())
            weights = list(top_countries.values())
            
            # Créer le graphique avec go.Figure comme pour les positions overlappées
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=weights,  # Utiliser directement la liste des poids
                y=countries,  # Utiliser directement la liste des pays
                orientation='h',
                marker_color='lightblue',
                text=[f"{w:.1f}%" for w in weights],  # Afficher les valeurs sur les barres
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Top 10 des Pays (% du portefeuille)",
                xaxis_title="% du portefeuille",
                yaxis_title="Pays",
                height=400,
                showlegend=False,
                # Inverser l'ordre pour avoir le plus gros en haut
                yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(countries))}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée pays disponible")
    
    with col2:
        
        # Préparer les données secteurs (top 10) - FILTRER Unknown et ignorés
        sector_data = portfolio_stats['sector_weights']
        filtered_sectors = {k: v for k, v in sector_data.items() 
                          if k not in ['Unknown', 'Ticker .XD ignoré'] and v > 0.1}
        top_sectors = dict(sorted(filtered_sectors.items(), key=lambda x: x[1], reverse=True)[:10])
        
        if top_sectors:
            # Extraire les données dans l'ordre
            sectors = list(top_sectors.keys())
            weights = list(top_sectors.values())
            
            # Créer le graphique avec go.Figure comme pour les positions overlappées
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=weights,  # Utiliser directement la liste des poids
                y=sectors,  # Utiliser directement la liste des secteurs
                orientation='h',
                marker_color='lightcoral',
                text=[f"{w:.1f}%" for w in weights],  # Afficher les valeurs sur les barres
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Top 10 des Secteurs (% du portefeuille)",
                xaxis_title="% du portefeuille",
                yaxis_title="Secteur",
                height=400,
                showlegend=False,
                # Inverser l'ordre pour avoir le plus gros en haut
                yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(sectors))}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée secteur disponible")

def render_top_holdings(portfolio_weights, holdings_data):
    """Affiche le top 30 des holdings du portefeuille"""
    st.subheader("📈 Top 30 des Holdings du Portefeuille")
    
    # Calculer le poids de chaque holding dans le portefeuille
    all_holdings = {}
    
    for etf_symbol, etf_weight in portfolio_weights.items():
        etf_data = holdings_data[holdings_data['ETF_Symbol'] == etf_symbol]
        
        # Grouper par ticker pour éviter les doublons
        for ticker in etf_data['Ticker'].unique():
            ticker_rows = etf_data[etf_data['Ticker'] == ticker]
            ticker_weight_in_etf = ticker_rows['Weight_Percent'].sum()  # Poids total du ticker dans l'ETF
            ticker_weight_in_portfolio = (ticker_weight_in_etf / 100) * (etf_weight / 100) * 100  # En pourcentage
            
            if ticker not in all_holdings:
                # Récupérer les infos de la première ligne pour ce ticker
                first_row = ticker_rows.iloc[0]
                all_holdings[ticker] = {
                    'ticker': ticker,
                    'company_name': first_row.get('Company_Name', ticker) or ticker,
                    'country': first_row.get('Country', 'Unknown') or 'Unknown',
                    'sector': first_row.get('Industry_Display', 'Unknown') or 'Unknown',
                    'weight_in_portfolio': 0,
                    'etfs': []
                }
            
            all_holdings[ticker]['weight_in_portfolio'] += ticker_weight_in_portfolio
            all_holdings[ticker]['etfs'].append({
                'etf': etf_symbol,
                'weight_in_etf': ticker_weight_in_etf
            })
    
    # Trier par poids décroissant et prendre le top 30
    top_holdings = sorted(all_holdings.values(), key=lambda x: x['weight_in_portfolio'], reverse=True)[:30]
    
    if top_holdings:
        # Créer le DataFrame
        holdings_df = pd.DataFrame([
            {
                'Rang': i + 1,
                'Ticker': holding['ticker'],
                'Pays': holding['country'],
                'Secteur': holding['sector'],
                'Poids Portefeuille (%)': round(holding['weight_in_portfolio'], 3),
                'Nb ETFs': len(holding['etfs']),
                'ETFs': ', '.join([etf['etf'] for etf in holding['etfs']])
            }
            for i, holding in enumerate(top_holdings)
        ])
        
        # Afficher le tableau avec une hauteur fixe pour le scroll
        st.dataframe(
            holdings_df,
            use_container_width=True,
            hide_index=True,
            height=600  # Hauteur fixe pour créer le scroll
        )
    
    else:
        st.info("Aucun holding trouvé")


######### GESTION DE LA SESSION (VERSION SESSION STATE SEULEMENT)
def generate_portfolio_export(portfolio_weights, available_etfs_info):
    """Génère le JSON d'export du portfolio pour téléchargement"""
    export_data = {
        'portfolio_name': f"Portfolio_ETF_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
        'created_date': pd.Timestamp.now().isoformat(),
        'total_etfs': len(portfolio_weights),
        'total_allocation': sum(portfolio_weights.values()),
        'portfolio': portfolio_weights,
        'etfs': []
    }
    
    for etf, weight in portfolio_weights.items():
        export_data['etfs'].append({
            'symbol': etf,
            'name': available_etfs_info.get(etf, 'Nom inconnu'),
            'allocation_percent': weight
        })
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def parse_portfolio_import(uploaded_file):
    """Parse un fichier JSON importé et extrait le portfolio"""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        data = json.loads(content)
        portfolio = {}
        
        if 'etfs' in data and isinstance(data['etfs'], list):
            for etf_data in data['etfs']:
                symbol = etf_data.get('symbol', '')
                allocation = etf_data.get('allocation_percent', 0)
                if symbol:
                    portfolio[symbol] = float(allocation)
        elif 'portfolio' in data:
            portfolio = data['portfolio']
        else:
            portfolio = data
        
        return portfolio, data.get('portfolio_name', 'Portfolio importé'), data.get('created_date', '')
    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'import: {e}")
        return {}, '', ''

def render_portfolio_management(available_etfs_info):
    """Gère l'export/import des portfolios (session state uniquement)"""
    st.markdown("##### 💾 Gestion du Portfolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export direct
        if st.button("💾 Exporter en JSON", help="Télécharge le portfolio actuel", key="export_portfolio"):
            if 'current_portfolio' in st.session_state and st.session_state['current_portfolio']:
                json_data = generate_portfolio_export(st.session_state['current_portfolio'], available_etfs_info)
                filename = f"portfolio_etf_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json"
                
                st.download_button(
                    label="⬇️ Télécharger le fichier JSON",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    key="download_portfolio"
                )
            else:
                st.warning("⚠️ Aucun portfolio à exporter")
    
    with col2:
        # Import JSON
        uploaded_file = st.file_uploader(
            "📁 Importer JSON",
            type=['json'],
            help="Importer un portfolio JSON",
            key="portfolio_uploader"
        )
        
        if uploaded_file is not None:
            imported_portfolio, portfolio_name, created_date = parse_portfolio_import(uploaded_file)
            
            if imported_portfolio:
                st.success(f"✅ Portfolio '{portfolio_name}' importé !")
                
                # Bouton pour appliquer
                if st.button("✅ Appliquer ce portfolio", key="apply_imported"):
                    st.session_state['portfolio_to_load'] = imported_portfolio
                    # Pas besoin de modifier num_etfs, on utilise maintenant etf_lines
                    st.rerun()

def convert_amounts_to_percentages(amounts):
    """Convertit les montants en euros en pourcentages"""
    total_amount = sum(amounts.values())
    if total_amount == 0:
        return {etf: 0.0 for etf in amounts.keys()}
    return {etf: (amount / total_amount) * 100 for etf, amount in amounts.items()}

###################################################################################################################
###################################################################################################################
###################################################################################################################

def main():
    # Appeler la fonction au début de main()
    load_custom_css()
    render_custom_header()

    # Charger la liste des ETFs disponibles
    st.header("🎯 Configuration du Portefeuille")
    
    with st.spinner("Chargement de la liste des ETFs..."):
        available_etfs_info = get_available_etfs()
    
    if not available_etfs_info:
        st.error("Impossible de charger la liste des ETFs. Vérifiez le fichier CSV.")
        return
    
    # Après avoir chargé available_etfs_info
    available_etfs = [""] + list(available_etfs_info.keys())

    render_portfolio_management(available_etfs_info)
    st.markdown("---")
    
    if 'portfolio_to_load' not in st.session_state:
        st.session_state['portfolio_to_load'] = {}

    if 'current_portfolio' not in st.session_state:
        st.session_state['current_portfolio'] = {}

    # NOUVEAU : Initialiser la liste des ETFs avec des IDs uniques
    if 'etf_lines' not in st.session_state:
        st.session_state['etf_lines'] = [0, 1]  # IDs uniques pour chaque ligne
        st.session_state['next_id'] = 2  # Prochain ID à utiliser

    # NOUVEAU : Sauvegarder les valeurs actuelles des widgets
    if 'saved_etf_values' not in st.session_state:
        st.session_state['saved_etf_values'] = {}
    if 'saved_weight_values' not in st.session_state:
        st.session_state['saved_weight_values'] = {}
    
    if 'saved_amount_values' not in st.session_state:
        st.session_state['saved_amount_values'] = {}

    # Charger le portfolio s'il y en a un
    portfolio_to_load = st.session_state.get('portfolio_to_load', {})

    # Si on charge un portfolio, créer les lignes nécessaires ET sauvegarder les valeurs
    if portfolio_to_load:
        needed_lines = max(len(portfolio_to_load), 2)
        if len(st.session_state['etf_lines']) < needed_lines:
            # Ajouter des lignes si nécessaire
            while len(st.session_state['etf_lines']) < needed_lines:
                st.session_state['etf_lines'].append(st.session_state['next_id'])
                st.session_state['next_id'] += 1
        
        # Sauvegarder les valeurs du portfolio importé
        etfs_list = list(portfolio_to_load.keys())
        for line_index, line_id in enumerate(st.session_state['etf_lines']):
            if line_index < len(etfs_list):
                etf_symbol = etfs_list[line_index]
                weight = portfolio_to_load[etf_symbol]
                st.session_state['saved_etf_values'][line_id] = etf_symbol
                st.session_state['saved_weight_values'][line_id] = weight

        # Ajouter un sélecteur de mode de saisie
    input_mode = st.radio(
        "Mode de saisie :",
        options=["Pourcentages (%)", "Montants (€)"],
        horizontal=True,
        key="input_mode"
    )

    # Initialiser les dictionnaires pour stocker les valeurs
    if 'saved_amount_values' not in st.session_state:
        st.session_state['saved_amount_values'] = {}

    portfolio_weights = {}
    portfolio_amounts = {}
    selected_etfs = []
    total_weight = 0
    total_amount = 0

    # Créer les colonnes d'en-tête
    col_header1, col_header2, col_header3 = st.columns([4, 1.5, 0.5])
    with col_header1:
        st.write("**Fonds**")
    with col_header2:
        if input_mode == "Pourcentages (%)":
            st.write("**Allocation** (%)")
        else:
            st.write("**Montant** (€)")

    # Itérer sur les IDs de lignes au lieu d'indices
    for line_index, line_id in enumerate(st.session_state['etf_lines']):
        col1, col2, col3 = st.columns([4, 1.5, 0.5])
        
        # Déterminer les valeurs par défaut depuis saved_values ou portfolio_to_load
        default_etf = st.session_state['saved_etf_values'].get(line_id, "")
        default_weight = st.session_state['saved_weight_values'].get(line_id, 0.0)
        default_amount = st.session_state['saved_amount_values'].get(line_id, 0.0)
        
        with col1:
            selected_etf = st.selectbox(
                f"ETF {line_index+1}",
                options=available_etfs,
                index=available_etfs.index(default_etf) if default_etf in available_etfs else 0,
                format_func=lambda x: f"{x} - {available_etfs_info.get(x, 'N/A')}" if x else "-- Sélectionnez un ETF --",
                key=f"etf_select_{line_id}",
                label_visibility="collapsed"
            )
        
        with col2:
            if input_mode == "Pourcentages (%)":
                value = st.number_input(
                    "Allocation (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight,
                    step=0.1,
                    key=f"weight_{line_id}",
                    disabled=not selected_etf,
                    label_visibility="collapsed"
                )
                weight = value
                amount = 0.0  # Pas utilisé en mode pourcentage
            else:
                value = st.number_input(
                    "Montant (€)",
                    min_value=0.0,
                    value=default_amount,
                    step=100.0,
                    key=f"amount_{line_id}",
                    disabled=not selected_etf,
                    label_visibility="collapsed"
                )
                amount = value
                weight = 0.0  # Sera calculé plus tard

        # SAUVEGARDER les valeurs actuelles
        st.session_state['saved_etf_values'][line_id] = selected_etf
        if input_mode == "Pourcentages (%)":
            st.session_state['saved_weight_values'][line_id] = weight
        else:
            st.session_state['saved_amount_values'][line_id] = amount

        with col3:
            # Afficher le bouton de suppression seulement s'il y a plus de 2 lignes
            if len(st.session_state['etf_lines']) > 2:
                if st.button("🗑️", key=f"delete_{line_id}", help="Supprimer cette ligne"):
                    # Supprimer cette ligne spécifique de la liste
                    st.session_state['etf_lines'].remove(line_id)
                    
                    # Nettoyer les valeurs sauvegardées pour cette ligne
                    for key_prefix in ['saved_etf_values', 'saved_weight_values', 'saved_amount_values']:
                        if line_id in st.session_state.get(key_prefix, {}):
                            del st.session_state[key_prefix][line_id]
                    
                    # Nettoyer aussi les valeurs du session state pour les widgets
                    for widget_prefix in [f"etf_select_{line_id}", f"weight_{line_id}", f"amount_{line_id}"]:
                        if widget_prefix in st.session_state:
                            del st.session_state[widget_prefix]
                    
                    st.rerun()
        
        if selected_etf and selected_etf != "":
            if selected_etf in selected_etfs:
                st.error(f"⚠️ L'ETF {selected_etf} est déjà sélectionné sur une autre ligne !")
            else:
                if input_mode == "Pourcentages (%)":
                    portfolio_weights[selected_etf] = weight
                    total_weight += weight
                else:
                    portfolio_amounts[selected_etf] = amount
                    total_amount += amount
                selected_etfs.append(selected_etf)

    # Convertir les montants en pourcentages si nécessaire
    if input_mode == "Montants (€)" and portfolio_amounts:
        portfolio_weights = convert_amounts_to_percentages(portfolio_amounts)
        total_weight = sum(portfolio_weights.values())

    # Vider portfolio_to_load après l'avoir utilisé
    if portfolio_to_load:
        st.session_state['portfolio_to_load'] = {}

    # Sauvegarder le portfolio actuel dans session state
    st.session_state['current_portfolio'] = portfolio_weights
    
    # Bouton pour ajouter une ligne - MODIFIÉ
    col_add1, col_add2, col_add3 = st.columns([4, 1.5, 0.5])
    with col_add1:
        if st.button("+ Ajouter un fonds"):
            # Ajouter une nouvelle ligne avec un ID unique
            new_id = st.session_state['next_id']
            st.session_state['etf_lines'].append(new_id)
            st.session_state['next_id'] += 1
            
            # Initialiser les valeurs par défaut pour la nouvelle ligne
            st.session_state['saved_etf_values'][new_id] = ""
            st.session_state['saved_weight_values'][new_id] = 0.0
            
            st.rerun()
    
    # Afficher le total et vérification
    if portfolio_weights:
        st.write(f"**Total allocation : {total_weight:.1f}%**")
        if abs(total_weight - 100.0) > 0.01:
            st.warning(f"⚠️ Le total devrait être 100%")
        
        # Bouton pour calculer l'overlap (seulement si au moins 2 ETFs sélectionnés)
        if len(portfolio_weights) >= 2:
            if st.button("🔍 Analyser l'Overlap", type="primary"):
                with st.spinner(f"Chargement des holdings pour {len(selected_etfs)} ETFs..."):
                    # Charger seulement les données des ETFs sélectionnés
                    holdings_data = load_etf_holdings(selected_etfs)
                
                if holdings_data is not None and not holdings_data.empty:
                    
                    # Calculer l'overlap
                    with st.spinner("Calcul de l'overlap..."):
                        overlaps, ticker_details, etf_holdings = calculate_overlap(portfolio_weights, holdings_data)
                        
                        # Calculer les similarités par paire
                        pairwise_similarities = calculate_pairwise_similarity(portfolio_weights, etf_holdings)
                        # Calculer l'overlap relatif
                        relative_overlap = calculate_relative_overlap(portfolio_weights, pairwise_similarities)
                    
                    
                    if overlaps:
                        st.markdown("<h1 style='text-align: center;'>📈 Résultats de l'Analyse</h1>", unsafe_allow_html=True)
                        
                        # PREMIER : Affichage du taux d'overlap global avec indicateur graphique
                        st.subheader("📊 Taux d'Overlap Global")
                        
                        # Calculer le taux d'overlap "parlant" = taux × nombre d'ETFs
                        overlap_rate_intuitive = relative_overlap
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Affichage plus grand et centré de la métrique
                            #st.markdown("### Similarité du portefeuille")
                            st.markdown(f"# {overlap_rate_intuitive:.1f}%")
                            
                            # Code couleur basé sur le taux
                            if overlap_rate_intuitive < 5:
                                color = "🟢"
                                status = "Excellente diversification"
                            elif overlap_rate_intuitive < 15:
                                color = "🟡" 
                                status = "Bonne diversification"
                            elif overlap_rate_intuitive < 30:
                                color = "🟠"
                                status = "Diversification à améliorer"
                            else:
                                color = "🔴"
                                status = "Redondance élevée"
                            
                            st.markdown(f"{color} {status}")
                            st.markdown(f"**Interprétation :** {overlap_rate_intuitive:.1f}% de votre portefeuille n'apporte aucune diversification")
                        
                        with col2:
                            # Indicateur graphique de similarité
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = overlap_rate_intuitive,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Similarité"},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 5], 'color': "green"},      # Excellent
                                        {'range': [5, 15], 'color': "lightgreen"}, # Bon
                                        {'range': [15, 30], 'color': "orange"},    # À améliorer  
                                        {'range': [30, 100], 'color': "red"}       # Problématique
                                    ],
                                    'threshold': {
                                        'line': {'color': "darkred", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 80
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        etf_list = list(portfolio_weights.keys())

                        if len(etf_list) >= 3:
                            
                            # Créer une matrice colorée avec style
                            similarity_data = []
                            for i, etf_a in enumerate(etf_list):
                                row = []
                                for j, etf_b in enumerate(etf_list):
                                    if i == j:
                                        row.append(100.0)  # Diagonale = 100%
                                    elif (etf_a, etf_b) in pairwise_similarities:
                                        row.append(pairwise_similarities[(etf_a, etf_b)])
                                    elif (etf_b, etf_a) in pairwise_similarities:
                                        row.append(pairwise_similarities[(etf_b, etf_a)])
                                    else:
                                        row.append(0.0)
                                similarity_data.append(row)
                            
                            # Créer la heatmap avec échelle de couleurs cohérente avec la jauge
                            fig_matrix = go.Figure(data=go.Heatmap(
                                z=similarity_data,
                                x=etf_list,
                                y=etf_list,
                                colorscale=[
                                    [0.0, "green"],      # 0%
                                    [0.20, "green"],     # 20%
                                    [0.40, "lightgreen"], # 40%  
                                    [0.60, "orange"],    # 60%
                                    [1.0, "red"]         # 100%
                                ],
                                zmin=0,
                                zmax=100,
                                text=[[f"{val:.1f}%" for val in row] for row in similarity_data],
                                texttemplate="%{text}",
                                textfont={"size": 16, "color": "white", "family": "Arial Black"},
                                hovertemplate="ETF A: %{y}<br>ETF B: %{x}<br>Similarité: %{text}<extra></extra>",
                                showscale=True,
                                colorbar=dict(
                                    title="Similarité (%)",
                                    tickfont=dict(size=12)
                                )
                            ))
                            
                            fig_matrix.update_layout(
                                title="Similarité entre ETFs",
                                xaxis_title="ETF",
                                yaxis_title="ETF",
                                height=400,
                                font=dict(size=12)
                            )
                            
                            st.plotly_chart(fig_matrix, use_container_width=True)

                       # TROISIÈME : Top 10 des positions overlappées
                        overlapped_positions = [(ticker, weight) for ticker, weight in overlaps.items() 
                                              if len(ticker_details[ticker]['etfs']) > 1]
                        top_overlapped = sorted(overlapped_positions, key=lambda x: x[1], reverse=True)[:10]
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("🏆 Top 10 des Positions Overlappées")
                            # Après le calcul, ajouter :
                            for ticker, details in ticker_details.items():
                                if len(details['etfs']) > len(portfolio_weights):
                                    st.write(f"❌ {ticker}: {len(details['etfs'])} ETFs détectés mais seulement {len(portfolio_weights)} dans le portefeuille")
                                    for etf_info in details['etfs']:
                                        st.write(f"  - {etf_info}")
                            
                            if top_overlapped:
                                overlap_df = pd.DataFrame([
                                    {
                                        'Ticker': ticker,
                                        'Poids dans le Portefeuille (%)': round(weight * 100, 3),
                                        'Nb ETFs': len(ticker_details[ticker]['etfs'])
                                    }
                                    for ticker, weight in top_overlapped
                                ])
                                
                                st.dataframe(
                                    overlap_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("Aucune position overlappée trouvée.")
                        
                        with col2:
                            st.subheader("📊 Poids des Positions Overlappées")
                            
                            if top_overlapped:
                                # Extraire explicitement les données du DataFrame dans l'ordre
                                tickers = overlap_df['Ticker'].tolist()
                                weights = overlap_df['Poids dans le Portefeuille (%)'].tolist()
                                
                                # Créer le graphique avec les données explicites
                                fig = go.Figure()
                                
                                fig.add_trace(go.Bar(
                                    x=weights,  # Utiliser directement la liste des poids
                                    y=tickers,  # Utiliser directement la liste des tickers
                                    orientation='h',
                                    marker_color='lightblue',
                                    text=[f"{w:.3f}%" for w in weights],  # Afficher les valeurs sur les barres
                                    textposition='outside'
                                ))
                                
                                fig.update_layout(
                                    title="Positions Overlappées",
                                    xaxis_title="Poids dans le Portefeuille (%)",
                                    yaxis_title="Ticker",
                                    height=400,
                                    showlegend=False,
                                    # Inverser l'ordre pour avoir le plus gros en haut
                                    yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(tickers))}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                st.info("Aucune donnée à afficher.")

                        st.subheader("📋 Vue d'ensemble du portefeuille")
                        st.write("")
                        portfolio_stats = calculate_portfolio_stats(portfolio_weights, holdings_data)
                        render_portfolio_overview(portfolio_stats, portfolio_weights)
                        render_geographic_and_sector_analysis(portfolio_stats)

                        render_top_holdings(portfolio_weights, holdings_data)

                    else:
                        st.warning("Aucun overlap trouvé avec cette sélection d'ETFs.")
                else:
                    st.error("Aucune donnée trouvée pour les ETFs sélectionnés.")
        else:
            st.info("👆 Sélectionnez au moins 2 ETFs pour pouvoir analyser l'overlap.")
    else:
        st.info("👆 Configurez votre portefeuille en sélectionnant des ETFs.")

if __name__ == "__main__":
    main()