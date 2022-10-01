
"""
    Created By : Xarnact
    Date : 28-09-2022
    Qoutes : life is a poetry 

"""

"""
    Utilities For Numeric 
"""

import plotly.express as px


def drawTopNNumericColumn():
    ...

def drawScatterplot():
    ...

def drawViolinPlot():
    ...

def drawBoxPlot():
    ...

def drawHistogram(df, col):
    fig = px.histogram(df, x=col)
    return fig

def drawHeatmapConfusion(confusionMatrix, title, tickVals ,tickText) :
    fig = px.imshow(conf, text_auto=True, aspect= "auto")
    
    fig.update_layout(
        xaxis=dict(tickmode="array",
                   tickvals=tickVals,
                   ticktext=tickText, side="top"),
        yaxis=dict(tickmode="array",
                   tickvals=tickVals,
                   ticktext=tickText
                   ),
        title_text= title,
        xaxis_title = "Predicted",
        yaxis_title="Actual"
    )
    return fig


def drawCountPlot(df, col1, color, barmode="group",difColor=False,percentage=False,title='',dropna=True):
    '''
    jangan pake percentage untuk dropna=False soalnya masi bug
    '''
    
    if difColor:
        dfGroped = df.groupby([col1,color],dropna=dropna).size().reset_index(name="counts")
    else:
        dfGroped = df.groupby([col1],dropna=dropna).size().reset_index(name="counts")
    dfGroped[dfGroped[col1].isna()]=dfGroped[dfGroped[col1].isna()].fillna("NaN")
    if percentage:
        dfGroped['percentage']= df.groupby([col1,color]).size().groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()),2)).values
        dfGroped.columns = [col1,color,'counts','percentage']
        fig = px.bar(dfGroped, x=col1,y="counts", color=color, text=dfGroped['percentage'],barmode=barmode,title=title)
        fig.update_layout(
            height=500,
            width=800
        )
    else:
        fig = px.bar(dfGroped, x=col1,y="counts", color=color, text_auto=True,barmode=barmode,title=title)
        fig.update_layout(
            height=500,
            width=800
        )
    return fig

def drawHeatmapCrosstab(df, col1, col2 ):
    crosstab = pd.crosstab(df[col1], df[col2])
    print(crosstab)

    fig = px.imshow(crosstab, text_auto=True,
                    width=500,aspect="auto")
    
    return fig

def drawHeatmap(data , x_label, y_label, x_title):
    fig = px.imshow(data, labels=dict(x=x_title),
                   x = x_label, y= y_label,
                   text_auto=True, aspect="auto", color_continuous_scale=px.colors.qualitative.G10)
    fig.update_layout(
        width=600
    )
    fig.update_xaxes(side="top")
    return fig

def diagnoseSupervisedModel():
    ...

def diagnoseUnsupervisedModel():
    ...

