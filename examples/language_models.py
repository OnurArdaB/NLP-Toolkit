import sys,pickle
sys.path.insert(1, '../src/toolkit')
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

    LM3_MLE = toolkit.LanguageModel(Docs,model_type="MLE",ngram=3)
    sentence,perplexity = toolkit.generate_sentence(LM3_MLE,text="milli")   
    print(sentence,perplexity)

    LM3_KneserNeyInterpolated = toolkit.LanguageModel(Docs,model_type="KneserNeyInterpolated",ngram=3)
    sentence,perplexity = toolkit.generate_sentence(LM3_KneserNeyInterpolated,text="milli")
    print(sentence,perplexity)
