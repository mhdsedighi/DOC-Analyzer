'DOCAnalyzer' reads all documents (PDF,Word,Text,PowerPointt,Excel,...) in a folder and sends it to your favorite local A.I.
You can chat with A.I. about your documents.
You can switch between folders easily.
Image to Text (OCR) for PDFs is used when necessary.


How to use the DOC Analyzer APP:


1-Install Ollama from here https://ollama.com/download
	(if you are from Iran you need a VPN)

2-Add a Model (or more) to Ollama by CMD command. (you can select from the app later)
for example:

	ollama run qwen2.5
	or:
	ollama run deepseek-r1:1.5

(see all Models here: https://ollama.com/library)


3-Run the APP:

*** Make sure your VPN or Proxy software is Not Active!
*** Make sure Ollama in running. (Its usually run automatically, but you can type 'ollama' in CMD to make sure)

 Run DocAnalayzer.exe

4-Note for Scanned PDFs (non-formatted PDFs)
 DOCAnalyzer automatically uses OCR technology for extracting text from scanned PDFs, Although it is recommended that you use a professional OCR software instead.
	you have to:
		-Download and install Tesseract software from here: https://github.com/tesseract-ocr/tesseract/releases
		-While installing Tesseract, add any language that you want from "additional language data" (e.g. German, Russian, Persian, etc...). DOCAnalyzer will detect language 				from among them.
		-Make sure you set the correct installation folder of Tesseract in DOCAnalyzer option window. (the default path is C:\Program Files\Tesseract-OCR)


	   
------------------------
How to keep the APP updated
------------------------
I am actively developing this APP.
Check for the latest Update here:
https://github.com/mhdsedighi/DOC-Analyzer/releases/tag/latest

or close the APP and Click on 'Update_App' once in a while to get the latest .exe file.



--------------------------------------------------------------------------------------------
If you are an advanced user, you can Run (or build) the APP yourself, especially if you need executable file for Linux or macOS.

1-Insatll Anaconda ( from here: https://www.anaconda.com/download )


2-Set the environment:
cd <project-folder> (or just open cmd in project folder). Enter in CMD (do this every time the code is updated)
		conda env update --name da-env --file environment.yml --prune

3-Run the APP
get the latest version of the code from GitHub: (https://github.com/mhdsedighi/DOC-Analyzer/archive/refs/heads/master.zip)
*** Make sure your VPN or Proxy software is Not Active!
open a CMD in this Folder
Click on the 'Run' (for Windows) file or enter this command in CMD
		conda activate da-env
		python .\DOC_Analyzer.py

I am actively developing this code. So if you don't know how to work with Git, just click on 'Update_Code' once in a while to get the latest python code.


