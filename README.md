# NaiveBayes

This is a just-for-fun script that I have made to test some ideas.

* Short info:
In few words it tries to be a author predictor script, whether you came to test the cases I provide
or to put your own books and test it by yourself I expect this will be handy.

there is a folder called books, just put there every book you want to use, both the books that will
serve to train the model and the books you want to predict the author from.

The script comes with an example of usage that may help you in case you want to see what happens with
other books.


* Technical details:
  if you doesn't put in the model any book for a given author, do not expect that the test function will
  predict the author. In other words, the test function will just give as result one of authors that are
  already in the model.
  
  the script is not plain to have in the model more than 1 book per author, if you want to train the 
  model with more that one book per author, try to join the corresponding txt files in just one. it may work
  
  Notice that if the number of words of a book increases, less reward that book will have when a new word is tested, 
  because if the word is in the book it's probability will be lower and also if the word is not in the book you punish
  all the books equally in that scenario. I have trie changing the test function:
    * Punish every book proportional to the size of it's book: result in a bias that benefits the shorter books
    * Punish every book equally to the size of the biggest book: result in a bias that benefits the shorter books
    
  Every spins around this: * in what extent we should reward a book when it hit a word? 
                           * in what extent we should punish a book when it doesn't hit a word?
  
  So how to fix this bias when you wan't to put in the model books of different number of words? 
  I still don't find a good resolution to this problem. Stay tune.
  
  nevertheless this script should work very well with book of similar length.
    
  
   
