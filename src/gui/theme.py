"""
Dark theme styling for the dashboard
"""

import streamlit as st


def get_color_palette(theme='dark'):
    """Get color palette for charts."""
    if theme == 'light':
        return {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#00c853',
            'danger': '#ff6b6b',
            'warning': '#ffa726',
            'info': '#4ecdc4',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2',
            'background': '#ffffff',
            'card_bg': '#f8f9fa',
            'text': '#1a1a1a',
            'text_secondary': '#666666',
            'border': '#e0e0e0'
        }
    else:  # dark theme
        return {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#00d4ff',
            'danger': '#ff6b6b',
            'warning': '#ffd93d',
            'info': '#4ecdc4',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2',
            'background': '#000000',
            'card_bg': '#0a0a0a',
            'text': '#fafafa',
            'text_secondary': '#a0a0a0',
            'border': '#333333'
        }


def apply_light_theme():
    """Apply light theme styling to the dashboard."""
    colors = get_color_palette('light')
    
    st.markdown(f"""
    <style>
        /* Global styles */
        .main {{
            background-color: {colors['background']} !important;
        }}
        
        .stApp {{
            background-color: {colors['background']} !important;
        }}
        
        /* Sidebar */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {colors['card_bg']} 0%, {colors['background']} 100%);
            border-right: 1px solid {colors['border']};
        }}
        
        /* Buttons */
        .stButton button {{
            background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }}
        
        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {colors['card_bg']};
            border-radius: 10px;
            padding: 0.5rem;
            border: 1px solid {colors['border']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 8px;
            color: {colors['text_secondary']};
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: rgba(102, 126, 234, 0.1);
            color: {colors['primary']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            color: white !important;
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {colors['text']};
        }}
        
        h2 {{
            border-bottom: 2px solid {colors['primary']};
            padding-bottom: 0.5rem;
        }}
        
        /* Cards */
        .metric-card {{
            background: {colors['card_bg']};
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid {colors['border']};
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: {colors['primary']};
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }}
    </style>
    """, unsafe_allow_html=True)


def apply_dark_theme():
    """Apply dark theme styling to the dashboard."""
    colors = get_color_palette('dark')
    
    st.markdown(f"""
    <style>
        /* Main theme colors */
        :root {{
            --primary-color: {colors['primary']};
            --secondary-color: {colors['secondary']};
            --background-color: {colors['background']};
            --text-color: {colors['text']};
        }}
        
        /* Global styles */
        .main {{
            background-color: {colors['background']} !important;
        }}
        
        .stApp {{
            background-color: {colors['background']} !important;
        }}
        
        /* Sidebar */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {colors['card_bg']} 0%, {colors['background']} 100%);
        }}
        
        /* Metric cards */
        [data-testid="stMetricValue"] {{
            font-size: 2rem;
            font-weight: 700;
            color: {colors['success']};
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: 0.9rem;
            color: {colors['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Buttons */
        .stButton button {{
            background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: {colors['card_bg']};
            border-radius: 10px;
            padding: 0.5rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 8px;
            color: {colors['text_secondary']};
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: rgba(102, 126, 234, 0.1);
            color: {colors['primary']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            color: white !important;
        }}
        
        /* Select boxes */
        .stSelectbox {{
            border-radius: 8px;
        }}
        
        .stSelectbox > div > div {{
            background-color: {colors['card_bg']};
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px;
            color: {colors['text']};
        }}
        
        /* Data tables */
        .dataframe {{
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .dataframe thead tr th {{
            background: linear-gradient(135deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            padding: 1rem;
        }}
        
        .dataframe tbody tr {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            transition: background-color 0.2s;
        }}
        
        .dataframe tbody tr:hover {{
            background-color: rgba(102, 126, 234, 0.1);
        }}
        
        /* Info boxes */
        .stInfo {{
            background-color: rgba(78, 205, 196, 0.1);
            border-left: 4px solid {colors['info']};
            border-radius: 8px;
        }}
        
        .stWarning {{
            background-color: rgba(255, 217, 61, 0.1);
            border-left: 4px solid {colors['warning']};
            border-radius: 8px;
        }}
        
        .stSuccess {{
            background-color: rgba(0, 212, 255, 0.1);
            border-left: 4px solid {colors['success']};
            border-radius: 8px;
        }}
        
        .stError {{
            background-color: rgba(255, 107, 107, 0.1);
            border-left: 4px solid {colors['danger']};
            border-radius: 8px;
        }}
        
        /* Charts */
        .js-plotly-plot {{
            border-radius: 10px;
            background-color: {colors['card_bg']};
            padding: 1rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {colors['card_bg']};
            border-radius: 8px;
            font-weight: 600;
            color: {colors['text']};
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: rgba(102, 126, 234, 0.1);
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {colors['background']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {colors['gradient_start']} 0%, {colors['gradient_end']} 100%);
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {colors['primary']};
        }}        
        /* Ensure black everywhere */
        .block-container {{
            background-color: {colors['background']} !important;
        }}
        
        .element-container {{
            background-color: transparent !important;
        }}
        
        section[data-testid="stSidebar"] > div {{
            background-color: {colors['card_bg']} !important;
        }}        
        /* Custom metric cards */
        .metric-card {{
            background: linear-gradient(135deg, {colors['card_bg']} 0%, rgba(102, 126, 234, 0.1) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            border-color: {colors['primary']};
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {colors['text']};
        }}
        
        h2 {{
            border-bottom: 2px solid {colors['primary']};
            padding-bottom: 0.5rem;
        }}
    </style>
    """, unsafe_allow_html=True)
