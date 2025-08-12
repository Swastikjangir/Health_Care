import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AssociationRuleMining:
    def __init__(self):
        self.frequent_itemsets = None
        self.rules = None
        self.transaction_encoder = TransactionEncoder()
        
    def prepare_transaction_data(self, data, method='binary'):
        """Prepare data for association rule mining"""
        if method == 'binary':
            # Convert data to binary format (presence/absence)
            if isinstance(data, pd.DataFrame):
                # If data is already in the right format
                if data.dtypes.apply(lambda x: x in ['bool', 'int64', 'float64']).all():
                    return data.astype(bool)
                else:
                    # Convert categorical data to binary
                    return pd.get_dummies(data).astype(bool)
            else:
                raise ValueError("Data must be a pandas DataFrame")
        
        elif method == 'transactions':
            # Handle transaction data (list of lists)
            if isinstance(data, list):
                te_data = self.transaction_encoder.fit(data).transform(data)
                return pd.DataFrame(te_data, columns=self.transaction_encoder.columns_)
            else:
                raise ValueError("For transaction method, data must be a list of lists")
        
        else:
            raise ValueError("Method must be 'binary' or 'transactions'")
    
    def find_frequent_itemsets(self, data, min_support=0.1, algorithm='apriori'):
        """Find frequent itemsets using specified algorithm"""
        if algorithm == 'apriori':
            self.frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
        elif algorithm == 'fpgrowth':
            self.frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
        else:
            raise ValueError("Algorithm must be 'apriori' or 'fpgrowth'")
        
        # Sort by support
        self.frequent_itemsets = self.frequent_itemsets.sort_values('support', ascending=False)
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        print(f"Top 5 itemsets by support:")
        print(self.frequent_itemsets.head())
        
        return self.frequent_itemsets
    
    def generate_association_rules(self, frequent_itemsets=None, metric='confidence', 
                                 min_threshold=0.5, support_only=False):
        """Generate association rules from frequent itemsets"""
        if frequent_itemsets is None:
            if self.frequent_itemsets is None:
                raise ValueError("No frequent itemsets available. Run find_frequent_itemsets first.")
            frequent_itemsets = self.frequent_itemsets
        
        if support_only:
            # Return only support values
            return frequent_itemsets
        
        # Generate association rules
        self.rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        # Sort by the specified metric
        if metric in self.rules.columns:
            self.rules = self.rules.sort_values(metric, ascending=False)
        
        print(f"Generated {len(self.rules)} association rules")
        print(f"Top 5 rules by {metric}:")
        print(self.rules.head())
        
        return self.rules
    
    def filter_rules(self, rules=None, min_confidence=0.5, min_lift=1.0, 
                    min_support=0.01, max_length=None):
        """Filter association rules based on various criteria"""
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules available. Run generate_association_rules first.")
            rules = self.rules
        
        filtered_rules = rules.copy()
        
        # Filter by confidence
        if 'confidence' in filtered_rules.columns:
            filtered_rules = filtered_rules[filtered_rules['confidence'] >= min_confidence]
        
        # Filter by lift
        if 'lift' in filtered_rules.columns:
            filtered_rules = filtered_rules[filtered_rules['lift'] >= min_lift]
        
        # Filter by support
        if 'support' in filtered_rules.columns:
            filtered_rules = filtered_rules[filtered_rules['support'] >= min_support]
        
        # Filter by antecedent/consequent length
        if max_length is not None:
            filtered_rules = filtered_rules[
                (filtered_rules['antecedents'].apply(len) <= max_length) &
                (filtered_rules['consequents'].apply(len) <= max_length)
            ]
        
        print(f"Filtered to {len(filtered_rules)} rules")
        return filtered_rules
    
    def get_top_rules(self, rules=None, n=10, metric='confidence'):
        """Get top N rules by specified metric"""
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules available. Run generate_association_rules first.")
            rules = self.rules
        
        if metric not in rules.columns:
            raise ValueError(f"Metric '{metric}' not found in rules")
        
        return rules.nlargest(n, metric)
    
    def analyze_rule_patterns(self, rules=None):
        """Analyze patterns in association rules"""
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules available. Run generate_association_rules first.")
            rules = self.rules
        
        analysis = {}
        
        # Rule length analysis
        if 'antecedents' in rules.columns and 'consequents' in rules.columns:
            antecedent_lengths = rules['antecedents'].apply(len)
            consequent_lengths = rules['consequents'].apply(len)
            
            analysis['rule_lengths'] = {
                'antecedent_lengths': antecedent_lengths.describe().to_dict(),
                'consequent_lengths': consequent_lengths.describe().to_dict(),
                'total_lengths': (antecedent_lengths + consequent_lengths).describe().to_dict()
            }
        
        # Support and confidence distribution
        if 'support' in rules.columns:
            analysis['support_distribution'] = rules['support'].describe().to_dict()
        
        if 'confidence' in rules.columns:
            analysis['confidence_distribution'] = rules['confidence'].describe().to_dict()
        
        # Lift analysis
        if 'lift' in rules.columns:
            analysis['lift_distribution'] = rules['lift'].describe().to_dict()
        
        # Top items in antecedents and consequents
        if 'antecedents' in rules.columns:
            all_antecedents = []
            for items in rules['antecedents']:
                all_antecedents.extend(list(items))
            
            antecedent_counts = pd.Series(all_antecedents).value_counts()
            analysis['top_antecedents'] = antecedent_counts.head(10).to_dict()
        
        if 'consequents' in rules.columns:
            all_consequents = []
            for items in rules['consequents']:
                all_consequents.extend(list(items))
            
            consequent_counts = pd.Series(all_consequents).value_counts()
            analysis['top_consequents'] = consequent_counts.head(10).to_dict()
        
        return analysis
    
    def visualize_rules(self, rules=None, method='scatter', top_n=50):
        """Visualize association rules"""
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules available. Run generate_association_rules first.")
            rules = self.rules
        
        # Take top N rules for visualization
        rules_viz = rules.head(top_n)
        
        if method == 'scatter':
            if 'support' in rules_viz.columns and 'confidence' in rules_viz.columns:
                fig = px.scatter(
                    rules_viz, 
                    x='support', 
                    y='confidence',
                    size='lift' if 'lift' in rules_viz.columns else None,
                    color='lift' if 'lift' in rules_viz.columns else None,
                    title='Association Rules: Support vs Confidence',
                    labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
                )
                return fig
        
        elif method == 'network':
            # Create network graph of rules
            import networkx as nx
            
            G = nx.DiGraph()
            
            for _, rule in rules_viz.iterrows():
                antecedent = list(rule['antecedents'])[0] if rule['antecedents'] else 'None'
                consequent = list(rule['consequents'])[0] if rule['consequents'] else 'None'
                
                G.add_edge(antecedent, consequent, 
                          weight=rule.get('confidence', 0),
                          support=rule.get('support', 0))
            
            # Create network visualization
            pos = nx.spring_layout(G)
            
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(edge[2]['weight'])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'))
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                marker=dict(size=20, color='lightblue')))
            
            fig.update_layout(
                title='Association Rules Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            
            return fig
        
        elif method == 'heatmap':
            # Create heatmap of rule metrics
            if 'support' in rules_viz.columns and 'confidence' in rules_viz.columns:
                # Create pivot table for heatmap
                pivot_data = rules_viz.pivot_table(
                    values='confidence', 
                    index='antecedents', 
                    columns='consequents', 
                    aggfunc='mean'
                ).fillna(0)
                
                fig = px.imshow(
                    pivot_data,
                    title='Association Rules Heatmap',
                    labels=dict(x='Consequents', y='Antecedents', color='Confidence'),
                    color_continuous_scale='Blues'
                )
                return fig
        
        else:
            raise ValueError("Method must be 'scatter', 'network', or 'heatmap'")
    
    def export_rules(self, rules=None, format_type='csv', filename='association_rules'):
        """Export association rules to file"""
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules available. Run generate_association_rules first.")
            rules = rules
        
        if format_type == 'csv':
            # Convert frozensets to strings for CSV export
            export_rules = rules.copy()
            
            if 'antecedents' in export_rules.columns:
                export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            
            if 'consequents' in export_rules.columns:
                export_rules['consequents'] = export_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            export_rules.to_csv(f"{filename}.csv", index=False)
            print(f"Rules exported to {filename}.csv")
            
        elif format_type == 'json':
            # Convert frozensets to lists for JSON export
            export_rules = rules.copy()
            
            if 'antecedents' in export_rules.columns:
                export_rules['antecedents'] = export_rules['antecedents'].apply(lambda x: list(x))
            
            if 'consequents' in export_rules.columns:
                export_rules['consequents'] = export_rules['consequents'].apply(lambda x: list(x))
            
            export_rules.to_json(f"{filename}.json", orient='records', indent=2)
            print(f"Rules exported to {filename}.json")
            
        else:
            raise ValueError("Format type must be 'csv' or 'json'")
    
    def create_health_insights(self, rules, health_context=True):
        """Create human-readable health insights from association rules"""
        if health_context:
            insights = []
            
            for _, rule in rules.iterrows():
                antecedent = list(rule['antecedents'])[0] if rule['antecedents'] else 'Unknown'
                consequent = list(rule['consequents'])[0] if rule['consequents'] else 'Unknown'
                
                support = rule.get('support', 0)
                confidence = rule.get('confidence', 0)
                lift = rule.get('lift', 0)
                
                insight = f"When {antecedent} is present, {consequent} is also present "
                insight += f"({confidence:.1%} of the time). "
                insight += f"This pattern occurs in {support:.1%} of all cases "
                insight += f"and is {lift:.2f}x more likely than random chance."
                
                insights.append({
                    'antecedent': antecedent,
                    'consequent': consequent,
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'insight': insight
                })
            
            return insights
        else:
            return rules
    
    def find_health_patterns(self, data, min_support=0.05, min_confidence=0.6, 
                           min_lift=1.2, health_conditions=None):
        """Find specific health-related patterns"""
        if health_conditions is None:
            health_conditions = [
                'diabetes', 'heart_disease', 'hypertension', 'obesity',
                'high_cholesterol', 'smoking', 'alcohol', 'exercise'
            ]
        
        # Find frequent itemsets
        frequent_itemsets = self.find_frequent_itemsets(data, min_support=min_support)
        
        # Generate rules
        rules = self.generate_association_rules(frequent_itemsets, 'confidence', min_confidence)
        
        # Filter rules for health conditions
        health_rules = []
        for _, rule in rules.iterrows():
            antecedent = list(rule['antecedents'])[0] if rule['antecedents'] else ''
            consequent = list(rule['consequents'])[0] if rule['consequents'] else ''
            
            # Check if rule involves health conditions
            if any(condition in antecedent.lower() or condition in consequent.lower() 
                   for condition in health_conditions):
                if rule.get('lift', 0) >= min_lift:
                    health_rules.append(rule)
        
        health_rules_df = pd.DataFrame(health_rules)
        
        if len(health_rules_df) > 0:
            health_rules_df = health_rules_df.sort_values('lift', ascending=False)
            print(f"Found {len(health_rules_df)} health-related association rules")
        
        return health_rules_df
