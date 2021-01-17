import sys,pickle
sys.path.insert(1, '../src/')
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

    WE = toolkit.WordVectors(Docs,300,'cbow',5)

    example_tuple_list =[('fransa','paris'),
                      ('almanya','berlin'),
                      ('italya','roma'),
                      ('ispanya','madrid'),
                      ('hollanda','amsterdam'),
                      ('ingiltere','londra'),
                      ('türkiye','ankara')]
    example_tuple_test =('rusya','')

    toolkit.word_relationship(WE,example_tuple_list,example_tuple_test)