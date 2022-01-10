

from pathlib import Path
import pickle
from google.cloud import storage
import io
from nltk.corpus import wordnet as wn
 
BLOCK_SIZE = 1999998

client = storage.Client()
bucket = client.get_bucket('316465533_313300857')

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
    def __init__(self):
        self._open_files = {}


    def read(self, locs, folder_name, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                # self._open_files[f_name] = open(f"buckets/{folder_name}/postings_gcp/{f_name}", "rb")

                self._open_files[f_name] = io.BytesIO(bucket.get_blob(f"{folder_name}/{f_name}").download_as_string())
                
            f = self._open_files[f_name]
            
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)
  




TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        pass



    def posting_lists_iter(self, folder_name, tokens = None):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
       
        if tokens is not None:
            posting_locs = []
            for token in tokens:
                if token in self.posting_locs:
                    posting_locs.append((token, self.posting_locs[token]))

                            
                        



            
        else:
            posting_locs = self.posting_locs.items()

        reader = MultiFileReader()
        for w, locs in posting_locs:
           
            b = reader.read(locs, folder_name, self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            yield w, posting_list
      

  

   
    