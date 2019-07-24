import unittest
from nnvmt import parseArguments
from nnvmt import decideTool
from nnvmt import decideOutput
from nnvmt import parseHandler
from src.nnvmt_exceptions import FileExtenstionError
from src.nnvmt_exceptions import OutputFormatError
import os

#class that implements unit tests for our tool
class TestNNVMT(unittest.TestCase):
    #tests decide tool function
    def test_decideTool(self):
        #open the tests text file
        test_path=os.path.join(os.getcwd(),"tests/decide_tool_tests.txt")
        #toolNames=["Keras","keras","tensorflow","Tensorflow","Reluplex","reluplex","nnet","Sherlock","sherlock","mat","Matfile"]
        file = open(test_path, "r") 
        line=True
        try:
            while line:
                line=file.readline().strip("\n")
                tup=line.split(",")
                if len(tup)==2:
                    #get the toolname
                    toolname=tup[0]
                    #get the file path
                    path=tup[1]
                    #get the filename
                    filename=os.path.basename(path)
                    #assert that reluplex files end in .nnet
                    if toolname in ["Reluplex","reluplex","nnet"] and not (".nnet" in filename):
                        self.assertRaises(FileExtenstionError,decideTool,toolname,path)
                    elif toolname in ["Reluplex","reluplex","nnet"]:
                        self.assertIn(".nnet",filename)
                    #assert that sherlock files end in .txt or have no file extension
                    elif toolname in ["sherlock","Sherlock"] and not (".txt" in filename or len(filename.split("."))==1):
                        self.assertRaises(FileExtenstionError,decideTool,toolname,path)
                    elif toolname in ["sherlock","Sherlock"]:
                        self.assertTrue(".txt" in filename or len(filename.split("."))==1)
                    #assert that keras files end in .h5 
                    elif toolname in ["keras","Keras"] and not (".h5" in filename):
                        self.assertRaises(FileExtenstionError,decideTool,toolname,path)
                    elif toolname in ["keras","Keras"]:
                        self.assertIn(".h5",filename)
                    #assert that tensorflow files have a .meta file 
                    elif toolname in ["tensorflow","Tensorflow"] and not (".meta" in filename):
                        self.assertRaises(FileExtenstionError,decideTool,toolname,path)
                    elif toolname in ["tensorflow","Tensorflow"]:
                        self.assertIn(".meta",filename)
                    #assert that Matfiles have a .mat file extension 
                    elif toolname in ["mat","Matfile"] and not (".mat" in filename):
                        self.assertRaises(FileExtenstionError,decideTool,toolname,path)
                    elif toolname in ["Matfile","mat"]:
                        self.assertIn(".mat",filename)
                    #assert that any other filename throws a Name Error
                    else:
                        self.assertRaises(NameError,decideTool,toolname,path)
        finally:            
            file.close()

