import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import RobertaTokenizerFast, TrainingArguments, Trainer, senten
    from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
    import pandas as pd
    import altair as alt
    from wordcloud import WordCloud
    return WordCloud, alt, load_dataset, mo, pd


@app.cell
def _(load_dataset, pd):
    dataset = load_dataset(
        "fddemarco/pushshift-reddit-comments", split="train", streaming=True
    ).remove_columns(
        column_names=[
            "link_id",
            "subreddit_id",
            "id",
            "created_utc",
            "controversiality",
        ]
    )
    df = dataset.take(100000)
    df = pd.DataFrame(list(df))
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## EDA
    """)
    return


@app.cell
def _(alt, pd):
    def create_top_n_pie_chart(
        df: pd.DataFrame, subreddit_col="subreddit", count_col="value", top_n=10
    ):
        df = df.groupby(subreddit_col).size().reset_index(name=count_col)

        # 1. Sort and identify Top N
        df_sorted = df.sort_values(count_col, ascending=False).reset_index(
            drop=True
        )
        df_top_n = df_sorted.head(top_n).copy()

        # 2. Calculate the percentage
        total_sum = df_sorted[count_col].sum()
        df_top_n["percentage"] = (df_top_n[count_col] / total_sum) * 100

        # 3. Calculate the 'Others' group value and percentage
        others_value = total_sum - df_top_n[count_col].sum()
        if others_value > 0:
            df_others = pd.DataFrame(
                [
                    {
                        subreddit_col: "Others",
                        count_col: others_value,
                        "percentage": (others_value / total_sum) * 100,
                    }
                ]
            )
            df_final = pd.concat([df_top_n, df_others], ignore_index=True)
        else:
            df_final = df_top_n.copy()

        # 4. Create the Altair Pie Chart with percentage
        base = (
            alt.Chart(df_final)
            .encode(
                theta=alt.Theta("percentage", stack=True),
                color=alt.Color(field=subreddit_col, title="Subreddit"),
                tooltip=[subreddit_col, count_col, "percentage"],
            )
            .properties(
                title=f"Top {top_n} Subreddits by Comment Count (Percentage)"
            )
        )

        pie = base.mark_arc(outerRadius=120).encode(
            order=alt.Order("percentage", sort="descending")
        )

        text = base.mark_text(radius=140).encode(
            text=alt.Text("percentage:Q", format=".1f"),
            order=alt.Order("percentage", sort="descending"),
            color=alt.value("black"),
        )

        return (pie + text).interactive()
    return (create_top_n_pie_chart,)


app._unparsable_cell(
    r"""
    wdef create_subreddit_histogram(df: pd.DataFrame):
        df = df.groupby('subreddit').size().reset_index(name='value')
        histogram = alt.Chart(df).mark_bar().encode(
            x=alt.X('subreddit', title='Subreddit', sort='-y'),  # Sort by count
            y=alt.Y('value:Q', title='Number of Comments'),
            color=alt.Color('subreddit', title='Subreddit')
        ).properties(
            title='Comments Count per Subreddit'
        )
    
        return histogram
    """,
    name="_"
)


@app.cell
def _(df):
    # Length of comments
    df["body"].map(lambda comment: len(comment)).describe()
    return


@app.cell
def _(df):
    df.groupby("subreddit").size().to_frame("value").describe()
    return


@app.cell
def _(create_top_n_pie_chart, df):
    create_top_n_pie_chart(df=df, top_n=10)
    return


@app.cell
def _():
    return


@app.cell
def _(WordCloud):
    WordCloud(max_words=1000).generate()
    return


@app.cell
def _(df):
    df["score"].describe()
    return


@app.cell
def _(df):
    from bertopic import BERTopic
    bertopic = BERTopic(embedding_model="all-MiniLM-L6-v2")
    topics, prob = bertopic.fit_transform(df["body"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
