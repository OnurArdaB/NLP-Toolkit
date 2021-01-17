import sys,pickle
sys.path.insert(1,r'C:\Users\Onur Arda Bodur\Documents\GitHub\NLP-Toolkit\src')
import toolkit

def ReadDocs(path):
    """

    This function will return a list which includes documents as string format.

    Parameters:
    path: path for the pickle file we will provide.

    """
    with open(path,'rb') as P:
        Docs = pickle.load(P)
    return Docs
    
    
if __name__ == "__main__":


    list_of_documents_path = sys.argv[1]
    Docs=ReadDocs(list_of_documents_path)
    
    
    wordcloud_outputfile = "/src/outputs/wordcloud.png"
    toolkit.WordCloud(Docs,8,wordcloud_outputfile,mode="TFIDF",stopwords=True)
    print("WordCloud function worked!")

    zips_outputfile = "/src/outputs/zips.png"
    toolkit.ZiphsPlot(Docs,zips_outputfile)
    print("Ziph's Law function worked!")

    heaps_outputfile = "/src/outputs/heaps.png"
    toolkit.HeapsPlot(Docs,heaps_outputfile)
    print("Heaps' Law function worked!")
