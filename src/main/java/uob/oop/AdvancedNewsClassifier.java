package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks
        for (int i = 0; i < Toolkit.getListVocabulary().size(); i++) {
            String currentVocabWord = Toolkit.getListVocabulary().get(i);
            if (!stopWordChecker(currentVocabWord)) {
                Vector currentVector = new Vector(Toolkit.listVectors.get(i));
                Glove objectGlove = new Glove(currentVocabWord, currentVector);
                listResult.add(objectGlove);
            }
        }
        return listResult;
    }

    public boolean stopWordChecker(String givenWord) {
        for (String currentStopWord : Toolkit.STOPWORDS) {
            if (givenWord.equalsIgnoreCase(currentStopWord)) {
                return true;
            }
        }
        return false;
    }

    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks

        //gets length of news content
        ArrayList<Integer> wordCountOfDocuments = new ArrayList<>();
        for (int i = 0; i < _listEmbedding.size(); i++) {
            int docWordCount = 0;
            String currentEmbeddingContent = _listEmbedding.get(i).getNewsContent();
            String[] embeddingToPieces = currentEmbeddingContent.split(" ");
            for (String currentWord : embeddingToPieces) {
                if (doesWordMatchGlove(currentWord)) {
                    docWordCount = docWordCount + 1;
                }
            }
            wordCountOfDocuments.add(docWordCount);
        }

        //sorts list in ascending order
        wordCountOfDocuments.sort(null);

        //calculates median
        int docSize = wordCountOfDocuments.size();
        if (docSize % 2 == 0) {
            int firstMedian = wordCountOfDocuments.get(docSize / 2);
            int secondMedian = wordCountOfDocuments.get(docSize / 2 + 1);
            intMedian = (firstMedian + secondMedian) / 2;
        } else {
            intMedian = (wordCountOfDocuments.get(docSize + 1)) / 2;
        }


        return intMedian;
    }

    public static boolean doesWordMatchGlove(String givenWord) {
        for (Glove currentGlove : AdvancedNewsClassifier.listGlove) {
            if (currentGlove.getVocabulary().equalsIgnoreCase(givenWord)) {
                return true;
            }
        }
        return false;
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (ArticlesEmbedding bedCurrent : this.listEmbedding) {
            try {
                bedCurrent.getEmbedding();
            } catch (InvalidSizeException catchException) {
                bedCurrent.setEmbeddingSize(embeddingSize);
            } catch (InvalidTextException catchException) {
                bedCurrent.getNewsContent();
            } catch (Exception exec) {
                throw new RuntimeException(exec);
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;
        //TODO Task 6.4 - 8 Marks
        for (ArticlesEmbedding currArticle : listEmbedding) {
            NewsArticles.DataType labelOfNews = currArticle.getNewsType();
            int numberOfLabel = Integer.parseInt(currArticle.getNewsLabel());
            if (labelOfNews.equals(NewsArticles.DataType.Training)) {
                inputNDArray = currArticle.getEmbedding();
                outputNDArray = Nd4j.zeros(1, _numberOfClasses);
                outputNDArray.putScalar(0, numberOfLabel - 1, 1);
                DataSet DSOfProject = new DataSet(inputNDArray, outputNDArray);
                listDS.add(DSOfProject);
            }
        }
        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        for (ArticlesEmbedding currentArticle : _listEmbedding) {
            NewsArticles.DataType labelOfNews = currentArticle.getNewsType();
            if (labelOfNews.equals(NewsArticles.DataType.Testing)) {
                int[] guessedLbleArray = myNeuralNetwork.predict(currentArticle.getEmbedding());
                int guessedLble = guessedLbleArray[0];
                listResult.add(guessedLble);
                currentArticle.setNewsLabel(String.valueOf(guessedLble));
            }
        }

        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks

        //creates list that holds all labels
        ArrayList<Integer> collectionOfTypes = new ArrayList<>();
        for (ArticlesEmbedding currentArticle : listEmbedding) {
            NewsArticles.DataType testingData = currentArticle.getNewsType();
            int labelToAdd = Integer.parseInt(currentArticle.getNewsLabel());
            if (!collectionOfTypes.contains(labelToAdd) && testingData.equals(NewsArticles.DataType.Testing)) {
                collectionOfTypes.add(labelToAdd);
            }
        }

        //sorts list from lowest to highest
        collectionOfTypes.sort(null);

        //prints output
        for (int dataLabel : collectionOfTypes) {
            int groupingLabel = dataLabel + 1;
            System.out.print("Group " + groupingLabel + "\r\n");
            for (ArticlesEmbedding currentArt : listEmbedding) {
                NewsArticles.DataType testingData = currentArt.getNewsType();
                int artLabel = Integer.parseInt(currentArt.getNewsLabel());
                int artGroupLabel = artLabel + 1;
                if (artGroupLabel == groupingLabel && testingData.equals(NewsArticles.DataType.Testing)) {
                    System.out.print(currentArt.getNewsTitle() + "\r\n");
                }
            }

        }
    }
}
