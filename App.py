import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go

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
        
        df = df.dropna(subset=['ETF_Symbol', 'Ticker', 'Weight_Percent'])
        df = df[df['Weight_Percent'] > 0]
        
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
    
    # Préparer la liste des options pour les selectbox
    available_etfs = [""] + list(available_etfs_info.keys())  # Option vide en premier
    
    # Interface directe : une ligne par ETF
    #st.subheader("Sélection des ETFs et Allocation")
    
    # Initialiser le nombre d'ETFs dans le state si pas encore fait
    if 'num_etfs' not in st.session_state:
        st.session_state['num_etfs'] = 2
    
    portfolio_weights = {}
    selected_etfs = []
    total_weight = 0
    
    # Créer les colonnes d'en-tête
    col_header1, col_header2, col_header3 = st.columns([4, 1.5, 0.5])
    with col_header1:
        st.write("**Fonds**")
    with col_header2:
        st.write("**Allocation** (%)")
    
    for i in range(st.session_state['num_etfs']):
        col1, col2, col3 = st.columns([4, 1.5, 0.5])
        
        with col1:
            # Sélecteur d'ETF avec option vide
            selected_etf = st.selectbox(
                f"ETF {i+1}",
                options=available_etfs,
                format_func=lambda x: f"{x} - {available_etfs_info.get(x, 'N/A')}" if x else "-- Sélectionnez un ETF --",
                key=f"etf_select_{i}",
                label_visibility="collapsed"
            )
        
        with col2:
            # Input pour l'allocation
            weight = st.number_input(
                "Allocation (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0 if not selected_etf else 100.0/st.session_state['num_etfs'],
                step=0.1,
                key=f"weight_{i}",
                disabled=not selected_etf,  # Désactivé si aucun ETF sélectionné
                label_visibility="collapsed"
            )
        
        with col3:
            # Bouton pour supprimer cette ligne (seulement si plus de 2 ETFs)
            if st.session_state['num_etfs'] > 2:
                if st.button("🗑️", key=f"delete_{i}", help="Supprimer cette ligne"):
                    st.session_state['num_etfs'] -= 1
                    st.rerun()
        
        # Si un ETF est sélectionné, l'ajouter au portefeuille
        if selected_etf and selected_etf != "":
            if selected_etf in selected_etfs:
                st.error(f"⚠️ L'ETF {selected_etf} est déjà sélectionné sur une autre ligne !")
            else:
                portfolio_weights[selected_etf] = weight
                selected_etfs.append(selected_etf)
                total_weight += weight
    
    # Bouton pour ajouter une ligne
    col_add1, col_add2, col_add3 = st.columns([4, 1.5, 0.5])
    with col_add1:
        if st.button("+ Ajouter un fonds"):
            st.session_state['num_etfs'] += 1
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
                            st.write("DEBUG - Détails pour vérification :")
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