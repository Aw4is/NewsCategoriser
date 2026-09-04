# News Classifier

A university Object-Oriented Programming assignment completed at the University of Birmingham.

The project builds a simple Java-based news classifier that processes a set of news articles and groups them by topic using introductory NLP and machine learning concepts. It was designed mainly to practise core OOP principles while working with text preprocessing, vector representations and similarity measures. 

## What it does

- Parses article titles and content from HTML
- Cleans and preprocesses text
- Builds a vocabulary from the article corpus
- Converts articles into TF-IDF vector representations
- Measures similarity using cosine similarity
- Groups related news articles based on their content

## Key Concepts

- Java and Object-Oriented Programming
- Separation of responsibilities across classes
- Text preprocessing and basic NLP
- TF-IDF embeddings
- Vector operations
- Cosine similarity
- Simple text classification
- JUnit testing

## Project Structure

The application is split into several core components:

- `HtmlParser` – extracts article titles and content
- `NLP` – handles text cleaning, lemmatisation and stop-word removal
- `Vector` – implements vector operations such as dot product and cosine similarity
- `AdvancedNewsClassifier` – builds TF-IDF representations and groups similar articles

## Outcome

The final program ranks articles by semantic similarity and groups them by topic using TF-IDF and cosine similarity. Through the project, I strengthened my understanding of OOP fundamentals, class responsibilities, vector-based text representation and basic NLP workflows.
