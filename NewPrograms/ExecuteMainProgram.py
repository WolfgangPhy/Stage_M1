import MainProgram
import CreateDataFile
import time

if __name__ == "__main__":
    start_time = time.time()
    
    mainprogram = MainProgram.MainProgram()
    mainprogram.run()
    
    #CreateDataFile = CreateDataFile.CreateDataFile()
    #CreateDataFile.execute()
    
    process_time = time.time() - start_time
    print("Process time: ", round(process_time, 0), " seconds")