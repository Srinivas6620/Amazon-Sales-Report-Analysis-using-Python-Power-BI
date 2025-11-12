import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
def load_data():
    df = pd.read_csv('C:/Users/teste/Downloads/Amazon Sale Report.csv', encoding='ISO-8859-1')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Order ID', 'Date', 'Amount'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Amount'])
    return df

df = load_data()

st.set_page_config(layout="wide", page_title="Amazon Sales Dashboard")
st.title("📊 Amazon Sales Analysis Dashboard")

# Sidebar menu
menu = [
    "📌 Project Overview",
    "📈 Sales Overview",
    "📦 Product Analysis",
    "🚚 Fulfillment Analysis",
    "📍 Geographical Analysis",
    "👥 Customer Segmentation",
    "🤖 Predictive Modeling"
]

choice = st.sidebar.radio("Select Analysis Section", menu)

if choice == "📌 Project Overview":
    st.subheader("Project Overview")
    st.markdown('''
    This project presents a detailed analysis of an Amazon Sales dataset.

    It covers several core business functions including sales performance tracking, product analysis, fulfillment effectiveness, customer segmentation, geographical insights, and predictive modeling. 

    The goal is to provide actionable insights for decision-making through visualizations and ML techniques:

    - Understand trends in revenue
    - Identify best-selling product categories
    - Analyze the success of different fulfillment methods
    - Segment customers by behavior
    - Detect geographical sales concentration
    - Predict order outcomes using machine learning
    
    Use the buttons on the left to explore different modules of the analysis.
    ''')

elif choice == "📈 Sales Overview":
    st.subheader("📈 Sales Performance Overview")
    monthly_sales = df.resample('M', on='Date')['Amount'].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_sales.plot(ax=ax)
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales Amount")
    st.pyplot(fig)

    st.markdown('''
    **Insights:**
    - Monthly sales trends show overall revenue performance.
    - Spikes or drops indicate promotional campaigns or seasonal impacts.
    - Useful for planning inventory and marketing efforts.
    - Helps forecast future demands based on trends.
    - Identifies months with underperformance for investigation.
    - The graph allows tracking YoY/MoM growth.
    - Useful for goal setting and benchmarking.
    - Sales dips may correlate with logistic or fulfillment issues.
    - Can align business strategy with performance cycles.
    - Acts as a foundation for revenue forecasting.
    ''')

elif choice == "📦 Product Analysis":
    st.subheader("📦 Product Category and Size Analysis")
    fig1, ax1 = plt.subplots()
    df['Category'].value_counts().head(10).plot(kind='barh', ax=ax1)
    ax1.set_title("Top Product Categories")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    df['Size'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title("Product Sizes Sold")
    st.pyplot(fig2)

    st.markdown('''
    **Insights:**
    - Shows which product categories dominate sales.
    - Guides marketing efforts towards top-performing items.
    - Helps remove or reconsider low-selling products.
    - Insights help in bundling popular sizes with products.
    - Helps in determining restocking and warehouse needs.
    - Can drive pricing strategy on popular items.
    - Reveals consumer preferences in terms of size.
    - Vital for planning promotional strategies.
    - Assists in new product development.
    - Supports supply chain optimization.
    ''')

elif choice == "🚚 Fulfillment Analysis":
    st.subheader("🚚 Fulfillment Method Effectiveness")
    fig1, ax1 = plt.subplots()
    df['Fulfilment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
    ax1.set_ylabel('')
    ax1.set_title("Fulfillment Methods")
    st.pyplot(fig1)

    fulfill_status = pd.crosstab(df['Fulfilment'], df['Status'])
    fig2, ax2 = plt.subplots()
    fulfill_status.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title("Order Status by Fulfillment Method")
    st.pyplot(fig2)

    st.markdown('''
    **Insights:**
    - Reveals which fulfillment types are most used.
    - Identifies which methods lead to successful deliveries.
    - Poor performing fulfillment types can be optimized or phased out.
    - Helps reduce cancellations and returns.
    - FBA (Fulfilled by Amazon) often has higher success rates.
    - Informs third-party logistics performance.
    - Supports decisions about warehouse locations.
    - Useful in SLA (service-level agreement) monitoring.
    - Helps in evaluating fulfillment costs.
    - Enhances customer satisfaction through timely delivery.
    ''')

elif choice == "📍 Geographical Analysis":
    st.subheader("📍 Geographical Sales Distribution")
    fig1, ax1 = plt.subplots()
    df.groupby('ship-city')['Amount'].sum().sort_values(ascending=False).head(10).plot(kind='bar', ax=ax1)
    ax1.set_title("Top Cities by Sales")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    df.groupby('ship-state')['Amount'].sum().sort_values(ascending=False).head(10).plot(kind='bar', color='green', ax=ax2)
    ax2.set_title("Top States by Sales")
    st.pyplot(fig2)

    st.markdown('''
    **Insights:**
    - Shows where demand is highest geographically.
    - Enables region-based marketing campaigns.
    - Helps optimize logistics and shipping hubs.
    - Informs stock distribution across warehouses.
    - Reveals untapped markets with low sales.
    - Supports franchise or physical store planning.
    - Informs regional pricing strategies.
    - Correlates customer location with product preference.
    - Helps manage delivery lead times by location.
    - Enables territory-based performance tracking.
    ''')

elif choice == "👥 Customer Segmentation":
    st.subheader("👥 Customer Segmentation using Clustering")
    customer_identifier ='ship-city'
    if customer_identifier in df.columns:
        customer_df = df.groupby(customer_identifier).agg({
            'Amount': 'sum',
            'Order ID': 'nunique'
        }).rename(columns={'Order ID': 'Total Orders'})

        scaler = StandardScaler()
        scaled = scaler.fit_transform(customer_df)

        kmeans = KMeans(n_clusters=3, random_state=42)
        customer_df['Segment'] = kmeans.fit_predict(scaled)

        silhouette = silhouette_score(scaled, customer_df['Segment'])

        fig, ax = plt.subplots()
        sns.scatterplot(data=customer_df, x='Amount', y='Total Orders', hue='Segment', palette='viridis', ax=ax)
        ax.set_title("Customer Segments")
        st.pyplot(fig)

        st.write(f"Silhouette Score: {silhouette:.3f}")

        st.markdown('''
        **Insights:**
        - Customers grouped by purchase volume and frequency.
        - High-value customers can be targeted for loyalty rewards.
        - Helps identify churn-prone or low-value customers.
        - Allows for personalized marketing strategies.
        - Segments can align with customer personas.
        - Drives decisions for tiered services.
        - Assists in customer support prioritization.
        - Improves targeting for upselling/cross-selling.
        - Helps monitor lifecycle of customer groups.
        - Supports product recommendations based on segments.
        ''')
    else:
        st.warning("Customer ID column is missing in dataset.")

elif choice == "🤖 Predictive Modeling":
    st.subheader("🤖 Predicting Order Status with ML")
    cat_cols = ['Category', 'Fulfilment', 'Sales Channel', 'ship-service-level']
    for col in cat_cols:
        df[col] = df[col].astype(str)

    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    y = encoder.fit_transform(df['Status'])
    X = df[['Category', 'Fulfilment','Amount']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    st.text("Classification Report:")
    st.text(report)

    st.markdown('''
    **Insights:**
    - Predicts whether an order will be Shipped, Cancelled, or Returned.
    - Random Forest used due to robustness and interpretability.
    - Accuracy indicates model reliability (over 80% is good).
    - Enables early interventions on risky orders.
    - Can alert logistics to possible delays or cancellations.
    - Helps identify key features influencing outcomes.
    - Model can be expanded with more features.
    - Drives automation in customer service handling.
    - Increases efficiency in order processing.
    - Supports decision making with data-driven foresight.
    ''')
