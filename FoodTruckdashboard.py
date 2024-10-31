import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Food Truck Menu Analysis Dashboard",
    page_icon="🚚",
    layout="wide"
)

class FoodTruckMenuAnalyzer:
    def __init__(self, df):
        self.df = df
        self.market_analysis = {}
        self.recommendations = {}
        
    def analyze_market(self):
        """
        Analyze existing market to identify patterns and opportunities
        """
        try:
            # 1. Category Distribution Analysis
            category_distribution = self.df['Category'].value_counts()
            category_percentage = (category_distribution / len(self.df) * 100).round(2)
            
            # 2. Popular Menu Items Analysis
            menu_frequency = self.df['Menu'].value_counts()
            
            # 3. Location-based Analysis
            location_items = self.df.groupby('Location')['Menu'].agg(list)
            
            # 4. Category Combinations
            truck_categories = self.df.groupby('Name')['Category'].agg(set)
            common_category_combinations = Counter([tuple(sorted(cats)) for cats in truck_categories])
            
            self.market_analysis = {
                'category_distribution': category_distribution.to_dict(),
                'category_percentage': category_percentage.to_dict(),
                'popular_items': menu_frequency.to_dict(),
                'location_items': location_items.to_dict(),
                'common_category_combinations': {str(k): v for k, v in common_category_combinations.items()}
            }
            
            return self.market_analysis
        except Exception as e:
            st.error(f"Error in market analysis: {str(e)}")
            return {}

    def identify_gaps(self):
        """
        Identify market gaps and opportunities
        """
        try:
            # 1. Underserved Categories
            category_saturation = pd.Series(self.market_analysis['category_percentage'])
            underserved_categories = category_saturation[category_saturation < 15].index.tolist()
            
            # 2. Unique Combinations
            existing_combinations = set(eval(k) for k in self.market_analysis['common_category_combinations'].keys())
            all_categories = set(self.df['Category'].unique())
            
            # Generate potential new combinations
            potential_combinations = []
            for i in range(2, 4):  # Try combinations of 2-3 categories
                for combo in combinations(all_categories, i):
                    if combo not in existing_combinations:
                        potential_combinations.append(combo)
                        
            self.market_analysis['gaps'] = {
                'underserved_categories': underserved_categories,
                'potential_category_combinations': potential_combinations[:5]
            }
            
            return self.market_analysis['gaps']
        except Exception as e:
            st.error(f"Error in gap analysis: {str(e)}")
            return {}

    def generate_menu_recommendations(self):
        try:
            # 1. Calculate optimal category mix
            recommended_categories = []
            for category, percentage in self.market_analysis['category_percentage'].items():
                if percentage < 20:
                    recommended_categories.append((category, 'High priority - underserved market'))
                elif percentage < 40:
                    recommended_categories.append((category, 'Medium priority - competitive market'))
                else:
                    recommended_categories.append((category, 'Low priority - saturated market'))
            
            # 2. Identify popular items
            total_trucks = len(self.df['Name'].unique())
            menu_suggestions = {
                'core_items': [],
                'differentiators': [],
                'innovative_combinations': []
            }
            
            # Add items that appear in 20-40% of trucks (popular but not oversaturated)
            for item, count in self.market_analysis['popular_items'].items():
                item_percentage = (count / total_trucks) * 100
                if 20 <= item_percentage <= 40:
                    menu_suggestions['core_items'].append((item, count))
            
            self.recommendations = {
                'recommended_categories': dict(recommended_categories),
                'menu_suggestions': menu_suggestions,
                'suggested_menu_size': {
                    'min': max(8, total_trucks // 2),
                    'max': max(15, total_trucks)
                }
            }
            
            return self.recommendations
        except Exception as e:
            st.error(f"Error in menu recommendations: {str(e)}")
            return {}

    def generate_report(self):
        try:
            self.analyze_market()
            self.identify_gaps()
            self.generate_menu_recommendations()
            
            report = {
                'market_overview': {
                    'total_trucks_analyzed': len(self.df['Name'].unique()),
                    'total_menu_items_analyzed': len(self.df),
                    'average_menu_size': len(self.df) / len(self.df['Name'].unique())
                },
                'market_analysis': self.market_analysis,
                'recommendations': self.recommendations
            }
            
            return report
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return {}

def load_data():
    try:
        # Try to read Food.csv directly
        df = pd.read_csv('Food.csv')
        return df
    except Exception as e:
        st.error(f"Error reading Food.csv: {str(e)}")
        # Fallback to file uploader
        uploaded_file = st.file_uploader("Upload your menu data CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            return df
        return None

def create_category_distribution_chart(data):
    category_counts = data['Category'].value_counts()
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title="Menu Category Distribution",
        labels={'x': 'Category', 'y': 'Number of Items'},
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    return fig

def create_popular_items_chart(data):
    item_counts = data['Menu'].value_counts().head(20)
    fig = px.bar(
        x=item_counts.index,
        y=item_counts.values,
        title="Top 20 Most Common Menu Items",
        labels={'x': 'Menu Item', 'y': 'Frequency'},
        color=item_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    return fig

def main():
    st.title("🚚 Food Truck Menu Analysis Dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ["Market Overview", "Menu Analysis", "Recommendations"]
    )
    
    # Load data
    df = load_data()
    
    if df is not None:
        analyzer = FoodTruckMenuAnalyzer(df)
        report = analyzer.generate_report()
        
        if page == "Market Overview":
            st.header("Market Overview")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Food Trucks", report['market_overview']['total_trucks_analyzed'])
            with col2:
                st.metric("Total Menu Items", report['market_overview']['total_menu_items_analyzed'])
            with col3:
                st.metric("Average Menu Size", f"{report['market_overview']['average_menu_size']:.1f}")
            
            # Category distribution
            st.subheader("Category Distribution")
            category_chart = create_category_distribution_chart(df)
            st.plotly_chart(category_chart, use_container_width=True)
            
            # Popular items
            st.subheader("Popular Menu Items")
            popular_items_chart = create_popular_items_chart(df)
            st.plotly_chart(popular_items_chart, use_container_width=True)
            
        elif page == "Menu Analysis":
            st.header("Menu Analysis")
            
            # Category details
            st.subheader("Category Analysis")
            for category, percentage in report['market_analysis']['category_percentage'].items():
                with st.expander(f"{category} ({percentage:.1f}%)"):
                    category_items = df[df['Category'] == category]['Menu'].unique()
                    st.write("Menu items in this category:", ", ".join(category_items))
            
            # Location analysis
            st.subheader("Location Analysis")
            locations = df['Location'].unique()
            selected_location = st.selectbox("Select a location", locations)
            if selected_location:
                location_items = df[df['Location'] == selected_location]['Menu'].unique()
                st.write(f"Menu items at {selected_location}:", ", ".join(location_items))
            
        elif page == "Recommendations":
            st.header("Recommendations for New Food Truck")
            
            # Menu size recommendation
            st.subheader("Recommended Menu Size")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Minimum Items", report['recommendations']['suggested_menu_size']['min'])
            with col2:
                st.metric("Maximum Items", report['recommendations']['suggested_menu_size']['max'])
            
            # Category recommendations
            st.subheader("Category Priority Recommendations")
            for category, priority in report['recommendations']['recommended_categories'].items():
                st.write(f"**{category}**: {priority}")
            
            # Menu suggestions
            st.subheader("Recommended Menu Items")
            if report['recommendations']['menu_suggestions']['core_items']:
                st.write("**Popular Items to Consider:**")
                for item, count in report['recommendations']['menu_suggestions']['core_items']:
                    st.write(f"- {item} (appears in {count} locations)")
            
            # Market gaps
            st.subheader("Market Opportunities")
            if 'gaps' in report['market_analysis']:
                st.write("**Underserved Categories:**")
                for category in report['market_analysis']['gaps']['underserved_categories']:
                    st.write(f"- {category}")

if __name__ == "__main__":
    main()