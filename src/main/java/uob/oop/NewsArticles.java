package uob.oop;

public class NewsArticles {
    public enum DataType {
        Training, Testing
    }

    private String newsTitle = "", newsContent = "";
    private DataType newsType = DataType.Testing;
    private String newsLabel = "-1";

    public NewsArticles(String _title, String _content, DataType _type, String _label) {
        this.newsTitle = _title;
        this.newsContent = _content;
        this.newsType = _type;
        this.newsLabel = _label;

    }

    public String getNewsLabel() {
        //TODO Task 2.2 - 0.5 marks
        return this.newsLabel; //Please modify the return value.
    }

    public DataType getNewsType() {
        //TODO Task 2.3 - 0.5 marks
        return this.newsType; //Please modify the return value.
    }

    public String getNewsTitle() {
        //TODO Task 2.4 - 0.5 marks
        return this.newsTitle; //Please modify the return value.
    }

    public String getNewsContent() {
        //TODO Task 2.5 - 0.5 marks
        return this.newsContent; //Please modify the return value.
    }

    public void setNewsLabel(String _lable) {
        //TODO Task 2.6 - 0.5 marks
        this.newsLabel = _lable;
    }

    public void setNewsType (DataType _type){
        //TODO Task 2.7 - 0.5
        this.newsType = _type;
    }
}
