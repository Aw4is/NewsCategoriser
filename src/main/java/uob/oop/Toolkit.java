package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        //TODO Task 4.1 - 5 marks
        listVocabulary = new ArrayList<>();
        listVectors = new ArrayList<>();
        try {
            File nameOfCurrentFile = getFileFromResource(FILENAME_GLOVE);
            myReader = new BufferedReader(new FileReader(nameOfCurrentFile));
            String currentLine;
            while ((currentLine = myReader.readLine()) != null) {
                String[] currentLineToPieces = currentLine.split(",");
                listVocabulary.add(currentLineToPieces[0]);
                double[] representationOfVectors = new double[currentLineToPieces.length - 1];
                for (int i = 1; i < currentLineToPieces.length; i++) {
                    representationOfVectors[i - 1] = Double.parseDouble(currentLineToPieces[i]);
                }
                listVectors.add(representationOfVectors);
                }
            } catch (URISyntaxException | IOException e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }



    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks
        try {
            File newsFold = getFileFromResource("News");
            if (newsFold.exists()) {
                File[] fileOfNews = newsFold.listFiles();
                if (fileOfNews != null) {
                    Stream<File> streamToSortFiles = Stream.of(fileOfNews)
                            .sorted(Comparator.comparing(File::getName));
                    streamToSortFiles.forEach(fileCurr -> {
                        if (fileCurr.isFile() && fileCurr.getName().endsWith(".htm")) {
                            StringBuilder newsArticle = new StringBuilder();
                            try (BufferedReader newsReader = Files.newBufferedReader(fileCurr.toPath())) {
                                String currentLine;
                                while ((currentLine = newsReader.readLine()) != null) {
                                    newsArticle.append(currentLine).append("\n");
                                }
                                String allOfNewsArticle = newsArticle.toString();
                                String currNewsArticleTitle = HtmlParser.getNewsTitle(allOfNewsArticle);
                                String currNewsArticleContent = HtmlParser.getNewsContent(allOfNewsArticle);
                                NewsArticles.DataType currNewsArticleType = HtmlParser.getDataType(allOfNewsArticle);
                                String currNewsArticleLabel = HtmlParser.getLabel(allOfNewsArticle);
                                NewsArticles currentNewsArticle = new NewsArticles(currNewsArticleTitle,
                                        currNewsArticleContent, currNewsArticleType, currNewsArticleLabel);
                                listNews.add(currentNewsArticle);
                            } catch (IOException currentException) {
                                throw new RuntimeException(currentException.getMessage());
                            }
                        }
                    });
                }
            }
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        return listNews;
    }



    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
