import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from scipy.stats import norm, shapiro
import BVCscrap as bvc
from datetime import date
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions
from pypfopt import plotting
import cvxpy as cp

# Configuration de la page
st.set_page_config(layout="wide", page_title="Advanced Stock Analysis & Portfolio Optimization")

# Liste des actions disponibles
available_actions = ['Addoha', 'AFMA', 'Afric Indus', 'Afriquia Gaz', 'Akdital', 'Alliances', 'Aradei Capital', 
                     'ATLANTASANAD', 'Attijariwafa', 'Auto Hall', 'Auto Nejma', 'BALIMA', 'BOA', 'BCP', 'BMCI', 
                     'Cartier Saada', 'CDM', 'CFG', 'CIH', 'Ciments Maroc', 'CMT', 'Colorado', 'COSUMAR', 'CTM', 
                     'Dari Couspate', 'Delta Holding', 'Disty Technolog', 'DISWAY', 'Ennakl', 'EQDOM', 'FENIE BROSSETTE',
                     'HPS', 'IBMaroc', 'Immr Invest', 'INVOLYS', 'Jet Contractors', 'LABEL VIE', 'LafargeHolcim', 
                     'Lesieur Cristal', 'M2M Group', 'Maghreb Oxygene', 'Maghrebail', 'Managem', 'Maroc Leasing', 
                     'Maroc Telecom', 'Microdata', 'Mutandis', 'Oulmes', 'PROMOPHARM', 'Rebab Company', 'Res.Dar Saada', 
                     'Risma', 'S2M', 'Sanlam Maroc', 'SALAFIN', 'SMI', 'Stokvis Nord Afr', 'SNEP', 'SODEP', 'Sonasid', 
                     'SOTHEMA', 'SRM', 'Ste Boissons', 'STROC Indus', 'TAQA Morocco', 'TGCC', 'Total Maroc', 'Unimer', 
                     'Wafa Assur', 'Zellidja']

# Initialisation des sessions
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {}
if 'global_start_date' not in st.session_state:
    st.session_state.global_start_date = pd.to_datetime("2020-01-01")
if 'global_end_date' not in st.session_state:
    st.session_state.global_end_date = pd.to_datetime(date.today())

# Fonctions d'analyse des rendements
def calculate_returns(data):
    returns = pd.DataFrame()
    for col in data.columns:
        returns[f'{col}_daily_return'] = data[col].pct_change()
        returns[f'{col}_log_return'] = np.log(data[col]/data[col].shift(1))
        returns[f'{col}_cumulative_return'] = (1 + returns[f'{col}_daily_return']).cumprod() - 1
    return returns.dropna()

def calculate_statistics(returns):
    stats_df = pd.DataFrame()
    for col in returns.columns:
        if '_daily_return' in col:
            action = col.replace('_daily_return', '')
            stats_df[action] = [
                returns[col].mean() * 252,
                returns[col].std() * np.sqrt(252),
                returns[col].skew(),
                returns[col].kurt(),
                returns[col].max(),
                returns[col].min(),
                returns[col].quantile(0.05),
                returns[col].quantile(0.95),
                shapiro(returns[col])[0],
                shapiro(returns[col])[1]
            ]
    stats_df.index = ['Rendement Annuel', 'Volatilité Annuelle', 'Asymétrie', 'Kurtosis', 
                     'Max Daily Return', 'Min Daily Return', 'VaR 5%', 'VaR 95%', 
                     'Shapiro-Wilk Stat', 'Shapiro-Wilk p-value']
    return stats_df

def plot_distributions(returns, action):
    col = f'{action}_daily_return'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogramme et KDE
    sns.histplot(returns[col], kde=True, ax=ax1, stat='density')
    ax1.set_title(f'Distribution des rendements - {action}')
    
    # QQ Plot
    stats.probplot(returns[col], dist="norm", plot=ax2)
    ax2.set_title(f'QQ Plot - {action}')
    
    # Boxplot
    sns.boxplot(x=returns[col], ax=ax3)
    ax3.set_title(f'Boxplot des rendements - {action}')
    
    st.pyplot(fig)

# Fonctions d'analyse de portefeuille
def monte_carlo_simulation(returns, num_portfolios=100000):
    preturns = []
    pvolatility = []
    pweights = []
    sharpe_ratio = []
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)
    
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        annual_return = np.sum(mean_returns * weights) * 252
        annual_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        preturns.append(annual_return)
        pvolatility.append(annual_std)
        pweights.append(weights)
        sharpe_ratio.append(annual_return / annual_std)
    
    return pd.DataFrame({
        'Returns': preturns,
        'Volatility': pvolatility,
        'Sharpe': sharpe_ratio,
        'Weights': pweights
    })

def markowitz_optimization(returns):
    mu = expected_returns.mean_historical_return(returns)
    S = risk_models.sample_cov(returns)
    
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    
    # Optimisations
    raw_weights_max_sharpe = ef.max_sharpe()
    cleaned_weights_max_sharpe = ef.clean_weights()
    ret_max_sharpe, vol_max_sharpe, _ = ef.portfolio_performance()
    
    ef = EfficientFrontier(mu, S)
    raw_weights_min_vol = ef.min_volatility()
    cleaned_weights_min_vol = ef.clean_weights()
    ret_min_vol, vol_min_vol, _ = ef.portfolio_performance()
    
    return {
        'max_sharpe': {
            'weights': cleaned_weights_max_sharpe,
            'return': ret_max_sharpe,
            'volatility': vol_max_sharpe
        },
        'min_vol': {
            'weights': cleaned_weights_min_vol,
            'return': ret_min_vol,
            'volatility': vol_min_vol
        }
    }

def calculate_portfolio_metrics(weights, returns):
    portfolio_returns = (returns * weights).sum(axis=1)
    
    metrics = {
        'Annual Return': portfolio_returns.mean() * 252,
        'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
        'Sharpe Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
        'Sortino Ratio': (portfolio_returns.mean() * 252) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)),
        'Max Drawdown': (portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()).max(),
        'VaR 5%': portfolio_returns.quantile(0.05),
        'CVaR 5%': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()
    }
    
    return metrics

# Interface utilisateur
st.title("📊 Advanced Stock Analysis & Portfolio Optimization")

# Fenêtre d'analyse des actions
st.header("🔍 Analyse des Actions Individuelles")

selected_actions = st.multiselect(
    "Sélectionnez les actions à analyser",
    available_actions
)

if selected_actions:
    # Contrôle des dates avec mise à jour de la session
    col1, col2 = st.columns(2)
    with col1:
        new_start_date = st.date_input("Date de début", st.session_state.global_start_date)
    with col2:
        new_end_date = st.date_input("Date de fin", st.session_state.global_end_date)
    
    # Mettre à jour les dates globales si elles ont changé
    if (new_start_date != st.session_state.global_start_date) or (new_end_date != st.session_state.global_end_date):
        st.session_state.global_start_date = new_start_date
        st.session_state.global_end_date = new_end_date
        # Vider le portefeuille car les dates ont changé
        st.session_state.portfolio = []
        st.session_state.portfolio_data = {}
        st.warning("Les dates ont été modifiées. Le portefeuille a été réinitialisé.")
    
    # Charger les données avec vérification
    try:
        data = bvc.loadmany(selected_actions, start=new_start_date, end=new_end_date)
        if data.empty:
            st.error("Aucune donnée disponible pour la période sélectionnée")
            st.stop()
            
        returns = calculate_returns(data)
        
        for action in selected_actions:
            if action not in data.columns:
                st.warning(f"Aucune donnée disponible pour {action} dans la période sélectionnée")
                continue
                
            st.subheader(f"📈 Analyse de {action}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bouton de téléchargement
                st.download_button(
                    label=f"📥 Télécharger {action} (CSV)",
                    data=data[[action]].to_csv(),
                    file_name=f"cours_{action}{new_start_date}{new_end_date}.csv",
                    mime="text/csv",
                    key=f"download_{action}"
                )
                
                # Graphique des prix
                st.markdown(f"### Prix de {action} - <span style='color:red; font-weight:bold;'>{data[action].iloc[-1]:.2f}</span>", 
                            unsafe_allow_html=True)
                fig = px.line(data, y=action, title=f"Évolution des prix - {action}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Dernières valeurs
                st.markdown("Dernières valeurs")
                st.dataframe(data[[action]].tail().style.format({action: "{:.2f}"}))
                
                # Variation quotidienne
                if len(data[action]) > 1:
                    current_price = data[action].iloc[-1]
                    prev_price = data[action].iloc[-2]
                    daily_change = ((current_price - prev_price) / prev_price) * 100
                    change_color = "red" if daily_change < 0 else "green"
                    st.markdown(
                        f"Variation: <span style='color:{change_color};'>{daily_change:+.2f}%</span>",
                        unsafe_allow_html=True
                    )
                
                # Rendements cumulés
                st.write(f"Rendements cumulés de {action}")
                cum_col = f'{action}_cumulative_return'
                fig = px.line(returns, y=cum_col, title=f"Rendements cumulés - {action}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statistiques descriptives
                st.write(f"Statistiques descriptives - {action}")
                stats_df = calculate_statistics(returns[[f'{action}_daily_return']])
                st.dataframe(stats_df.style.format({
                    'Rendement Annuel': '{:.2%}',
                    'Volatilité Annuelle': '{:.2%}',
                    'VaR 5%': '{:.2%}',
                    'VaR 95%': '{:.2%}',
                    'Shapiro-Wilk p-value': '{:.4f}'
                }))
                
                # Test de normalité
                shapiro_stat, shapiro_p = shapiro(returns[f'{action}_daily_return'])
                st.write(f"Test de normalité (Shapiro-Wilk):")
                st.write(f"Statistique: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
                if shapiro_p > 0.05:
                    st.success("Les rendements suivent une distribution normale (p > 0.05)")
                else:
                    st.error("Les rendements ne suivent pas une distribution normale (p ≤ 0.05)")
            
            # Graphiques de distribution
            st.write(f"Distribution des rendements - {action}")
            plot_distributions(returns, action)
            
            # Bouton d'ajout au portefeuille
            if st.button(f"✅ Ajouter {action} au portefeuille", key=f"add_{action}"):
                if action not in st.session_state.portfolio:
                    # Stocker les données spécifiques à cette action
                    action_data = {
                        'raw': data[[action]].copy(),
                        'returns': returns[[f'{action}_daily_return']].copy().rename(columns={f'{action}_daily_return': action})
                    }
                    
                    st.session_state.portfolio_data[action] = action_data
                    st.session_state.portfolio.append(action)
                    st.success(f"{action} ajoutée au portefeuille avec les données du {new_start_date} au {new_end_date}")
                else:
                    st.warning(f"{action} est déjà dans le portefeuille")

    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")

# Fenêtre de gestion de portefeuille
st.header("💰 Gestion du Portefeuille")

if len(st.session_state.portfolio) > 0:
    st.subheader("Portefeuille Actuel")
    st.write(f"Période analysée: {st.session_state.global_start_date} au {st.session_state.global_end_date}")
    st.write(st.session_state.portfolio)
    
    # Bouton de téléchargement du portefeuille complet
    if st.button("📥 Télécharger tous les cours du portefeuille (CSV)"):
        all_data = pd.concat([st.session_state.portfolio_data[action]['raw'] 
                            for action in st.session_state.portfolio], axis=1)
        csv = all_data.to_csv()
        st.download_button(
            label="Télécharger maintenant",
            data=csv,
            file_name=f"cours_portefeuille_{st.session_state.global_start_date}_{st.session_state.global_end_date}.csv",
            mime="text/csv"
        )
    
    # Construction des données du portefeuille
    portfolio_returns = pd.concat(
        [st.session_state.portfolio_data[action]['returns'] 
         for action in st.session_state.portfolio],
        axis=1
    ).dropna()
    
    valid_actions = [col for col in portfolio_returns.columns if not portfolio_returns[col].isnull().all()]
    
    if len(valid_actions) == 0:
        st.error("Aucune donnée de rendement valide disponible pour les actions du portefeuille")
        st.stop()
    
    if len(valid_actions) < len(st.session_state.portfolio):
        missing_actions = set(st.session_state.portfolio) - set(valid_actions)
        st.warning(f"Données manquantes pour les actions: {', '.join(missing_actions)}")
    
    # Matrice de corrélation
    st.subheader("Matrice de Corrélation")
    corr_matrix = portfolio_returns.corr()
    
    fig = px.imshow(
        corr_matrix,
        x=valid_actions,
        y=valid_actions,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        text_auto=True,
        title="Corrélation entre les rendements des actifs"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimisation seulement si au moins 2 actions valides
    if len(valid_actions) >= 2:
        st.subheader("Optimisation de Portefeuille")
        
        # Simulation Monte Carlo
        if st.button("Lancer la Simulation Monte Carlo"):
            with st.spinner('Simulation de 100,000 portefeuilles...'):
                mc_results = monte_carlo_simulation(portfolio_returns)
            
            # Affichage des résultats
            fig = px.scatter(
                mc_results, 
                x='Volatility', 
                y='Returns', 
                color='Sharpe',
                title='Frontière Efficiente - Simulation Monte Carlo'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Portefeuilles optimaux
            col1, col2 = st.columns(2)
            
            with col1:
                max_sharpe_idx = mc_results['Sharpe'].idxmax()
                st.markdown("Portefeuille Sharpe Maximum")
                st.metric("Rendement annuel", f"{mc_results.loc[max_sharpe_idx, 'Returns']:.2%}")
                st.metric("Volatilité annuelle", f"{mc_results.loc[max_sharpe_idx, 'Volatility']:.2%}")
                st.metric("Ratio de Sharpe", f"{mc_results.loc[max_sharpe_idx, 'Sharpe']:.2f}")
                
                weights = pd.Series(
                    mc_results.loc[max_sharpe_idx, 'Weights'], 
                    index=valid_actions
                )
                fig = px.pie(
                    values=weights*100, 
                    names=weights.index, 
                    title='Allocation Sharpe Max'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                min_vol_idx = mc_results['Volatility'].idxmin()
                st.markdown("Portefeuille Variance Minimum")
                st.metric("Rendement annuel", f"{mc_results.loc[min_vol_idx, 'Returns']:.2%}")
                st.metric("Volatilité annuelle", f"{mc_results.loc[min_vol_idx, 'Volatility']:.2%}")
                st.metric("Ratio de Sharpe", f"{mc_results.loc[min_vol_idx, 'Sharpe']:.2f}")
                
                weights = pd.Series(
                    mc_results.loc[min_vol_idx, 'Weights'], 
                    index=valid_actions
                )
                fig = px.pie(
                    values=weights*100, 
                    names=weights.index, 
                    title='Allocation Variance Min'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Optimisation Markowitz
        if st.button("Optimisation Markowitz"):
            try:
                with st.spinner('Optimisation en cours...'):
                    results = markowitz_optimization(portfolio_returns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("Portefeuille Sharpe Maximum")
                    st.metric("Rendement annuel", f"{results['max_sharpe']['return']:.2%}")
                    st.metric("Volatilité annuelle", f"{results['max_sharpe']['volatility']:.2%}")
                    st.metric("Ratio de Sharpe", 
                             f"{results['max_sharpe']['return'] / results['max_sharpe']['volatility']:.2f}")
                    
                    weights = pd.Series(results['max_sharpe']['weights'])
                    fig = px.pie(
                        values=weights*100, 
                        names=weights.index, 
                        title='Allocation Sharpe Max'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("Portefeuille Variance Minimum")
                    st.metric("Rendement annuel", f"{results['min_vol']['return']:.2%}")
                    st.metric("Volatilité annuelle", f"{results['min_vol']['volatility']:.2%}")
                    st.metric("Ratio de Sharpe", 
                             f"{results['min_vol']['return'] / results['min_vol']['volatility']:.2f}")
                    
                    weights = pd.Series(results['min_vol']['weights'])
                    fig = px.pie(
                        values=weights*100, 
                        names=weights.index, 
                        title='Allocation Variance Min'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Erreur lors de l'optimisation: {str(e)}")
        
        # Analyse de portefeuille personnalisé
        st.subheader("Analyse de Portefeuille Personnalisé")
        
        weights = {}
        cols = st.columns(len(valid_actions))
        for i, action in enumerate(valid_actions):
            with cols[i]:
                weights[action] = st.slider(
                    f"Poids {action}", 
                    0.0, 1.0, 
                    1.0/len(valid_actions),
                    key=f"weight_{action}"
                )
        
        # Normalisation des poids
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
            
            if st.button("Analyser ce portefeuille"):
                try:
                    metrics = calculate_portfolio_metrics(
                        pd.Series(weights), 
                        portfolio_returns
                    )
                    
                    # Affichage des métriques
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rendement Annuel", f"{metrics['Annual Return']:.2%}")
                        st.metric("Volatilité Annuelle", f"{metrics['Annual Volatility']:.2%}")
                    
                    with col2:
                        st.metric("Ratio de Sharpe", f"{metrics['Sharpe Ratio']:.2f}")
                        st.metric("Ratio de Sortino", f"{metrics['Sortino Ratio']:.2f}")
                    
                    with col3:
                        st.metric("VaR 5% (Quotid.)", f"{metrics['VaR 5%']:.2%}")
                        st.metric("CVaR 5% (Quotid.)", f"{metrics['CVaR 5%']:.2%}")
                    
                    # Graphiques
                    fig = px.line(
                        (1 + portfolio_returns.dot(pd.Series(weights))).cumprod(),
                        title="Performance Cumulative du Portefeuille"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig = px.histogram(
                        portfolio_returns.dot(pd.Series(weights)),
                        nbins=50,
                        title="Distribution des Rendements Quotidiens"
                    )
                    fig.add_vline(
                        x=metrics['VaR 5%'], 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="VaR 5%"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Erreur dans l'analyse: {str(e)}")
    else:
        st.warning("Au moins 2 actions avec données valides sont nécessaires pour l'optimisation")
else:
    st.warning("Veuillez ajouter des actions à votre portefeuille pour commencer l'analyse")