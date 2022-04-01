from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def display_percent_count_plot(ax, feature, rd):
    """
        Display percentual value of countings for each x value in
        a seaborn countplot.
        
        inputs: - ax (axis on which the count plot is drawn):              matplotlib.Axis
        ------- - feature (feature displayed on the x axis):               pandas.Series
                - rd (number of digits from the decimal point to display): int
        
        outputs: None
        -------
    """
    
    total = len(feature)
    for p in ax.patches:
        percentage = ("{:."+str(rd)+"f}%").format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 12)
        
    return None

## ---

def display_feat_import(X_values, y_values, ax):
    
    """
        Display a grid containing 4 2D plots regarding feature importance
        of a dichotomous classification problem. It uses three different
        algorithms: Logistice Regression, Random Forest, and PCA.
        
        Further, it displays the cumulative explained variance given by 
        the PCA algorithm, and gives the linear combination coefficients
        from the constructed principal components.
        
        inputs: - X_values (a 2D array containing features):        pandas.DataFrame
        ------- - y_values (a 1D array containing target classes):  pandas.Series
                - ax (a matplotlib axis grid to plot figures on):   matplotlib.axes._subplots
                
        outputs: - loadings (coefficients of the linear combination
        --------            of the original variables from which the 
                            principal components are constructed):   pandas.DataFrame
    """
        
    scaler = StandardScaler()
    models = [LogisticRegression(),
              RandomForestClassifier(n_estimators=200, max_depth=7),
              PCA()]
    
    for axis, model in zip(ax.reshape(-1)[:3], models):
        
            clf = make_pipeline(StandardScaler(), model)
            clf.fit(X_values, y_values)
            
            if clf.steps[1][0]=='pca':
                loadings = pd.DataFrame(
                    data=clf[1].components_.T * np.sqrt(clf[1].explained_variance_), 
                    columns=[f'PC{i}' for i in range(1, len(X.columns) + 1)],
                    index=X.columns
                )
                
                pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
                pc1_loadings = pc1_loadings.reset_index()
                pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

                axis.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
                axis.set_title('PCA loading scores (first principal component)', size=12)
                axis.set_xticklabels(pc1_loadings['Attribute'],rotation='vertical')
                
                ax[1][1].plot(clf[1].explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
                ax[1][1].set_title('Cumulative explained variance', size=12)
                ax[1][1].set_xlabel('Number of Principal Components')

            else:
                try:
                    importances = pd.DataFrame(data={
                        'Attribute': X.columns,
                        'Importance': clf[1].coef_[0]
                    })
                except:
                    importances = pd.DataFrame(data={
                        'Attribute': X.columns,
                        'Importance': clf[1].feature_importances_
                    })
                importances = importances.sort_values(by='Importance', ascending=False)

                axis.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
                axis.set_title('Feature importances from '+clf.steps[1][0], size=12)
                axis.set_xticklabels(importances['Attribute'],rotation='vertical')
            
    return loadings

## ---

def cf_matrix_labels(cf_matrix):
    """
        Produce labels from a 2x2 confusion to plot on a seaborn heatmap.
        
        inputs: - cf_matrix (2x2 confusion matrix):     numpy 2D array
        -------
        
        outputs: labels (a list containing the labels): numpy 1D array
        --------
    """
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    return labels

def dist_medians(df, feat, hue):
    """
        Compute the differences between medians of a given feture
        considering hue levels.
        
        inputs: - df (data) :                  pandas.DataFrame
        ------- - feat (feature name):         str
                - hue (name of hue variable ): str
                
        outputs: a pandas.Series containing the difference between medians
        --------
    """
    medians = [df[df[hue] == val][feat].median() for val in df[hue].unique()]
    return pd.Series(medians).diff()[1:]

