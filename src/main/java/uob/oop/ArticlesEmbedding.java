package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        this.intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        //Text cleaning
        if (!processedText.isEmpty()){
            return processedText;
        }
        String PurifiedText = textCleaning(super.getNewsContent());

        //Text Lemm
        Properties pipeLineProperties = new Properties();
        pipeLineProperties.setProperty("annotators", "tokenize,pos,lemma");
        StanfordCoreNLP pipe = new StanfordCoreNLP(pipeLineProperties);
        CoreDocument doc = pipe.processToCoreDocument(PurifiedText);

        //Removes stop words
        StringBuilder stopWordsRemovedText = new StringBuilder();
        for (CoreLabel currentToken : doc.tokens()){
        String currentLemma = currentToken.lemma();
        boolean encounteredStopWord = false;
        for (String currentStopWord : Toolkit.STOPWORDS){
            if (currentStopWord.equalsIgnoreCase(currentLemma)){
                encounteredStopWord = true;
                break;
            }
        }
        if (!encounteredStopWord){
            stopWordsRemovedText.append(currentLemma).append(" ");
        }
        }
        processedText = stopWordsRemovedText.toString();

        return processedText.trim();
    }


    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks

        //Throws errors + checks if already been processed
        if (!newsEmbedding.isEmpty()){
            return newsEmbedding;
        }
        if (intSize == -1){
            throw new InvalidSizeException("Invalid size");
        }
        if(processedText.isEmpty()){
            throw new InvalidTextException("Invalid text");
        }


        //Gets dimensions of ND4 Array + creates it
        int coordY = AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize();
        int coordX = intSize;
        String[] processedTextToPieces = processedText.split(" ");
        this.newsEmbedding = Nd4j.zeros(coordX,coordY);

        //gets embeddings
        int correctLength = Math.min(processedTextToPieces.length, intSize);
        int arrayRow = 0;
        for (String currentWord : processedTextToPieces) {
                Glove gloveToCheck = gloveCorrespondingToWord(currentWord);
                if (gloveToCheck != null) {
                    newsEmbedding.putRow(arrayRow++, Nd4j.create(gloveToCheck.getVector().getAllElements()));
                    if (arrayRow == correctLength) {
                        break;
                    }
                }
        }
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    public static Glove gloveCorrespondingToWord(String wordToCompare){
        for (Glove currGl : AdvancedNewsClassifier.listGlove){
            if (currGl.getVocabulary().equalsIgnoreCase(wordToCompare)){
                return currGl;
            }
        }
        return null;
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
