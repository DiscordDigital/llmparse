import ollama
import json
import random
import operator
import os
import sys
from textwrap import dedent
from tqdm import tqdm
from pathlib import Path
from time import sleep

# Define the LLMs used for the benchmarks
LLMs = ["llama3.2:3b", "gemma3:4b", "mistral:7b"]

# Function to check if all required models are available.
def check_models():
    """
    Verify that all required LLM models are available.

    This function checks if the necessary models are installed and provides instructions on how to obtain them if they are missing.
    """

    # Create a copy of the list of LLMs and append an additional model for consideration
    allModels = LLMs.copy()
    allModels.append("mistral:7b-instruct")

    # Retrieve the list of available models from ollama
    available_models_data = ollama.list()["models"]

    # Extract the model names from the available models data
    available_models = []
    for model in available_models_data:
        available_models.append(model.model)

    # Iterate over each required model
    for requiredModel in allModels:
        skip = False # Reset skip flag for each required model

        # Check if the required model is available
        for availableModel in available_models:
            # If the available model matches the required model, set skip to True
            if availableModel.startswith(requiredModel):
                skip = True 

        # If the required model was found, move on to the next one
        if skip:
            continue

        # If the model wasn't found, print an error message and provide instructions on how to obtain it
        print(f"{requiredModel} is missing. You can obtain it using the command: ollama pull {requiredModel}")

        # Exit the program with a non-zero status code to indicate an error
        exit(1)

# This will check if all required models are available on the system
check_models()

# Check if the user has provided an input file as a command-line argument
if len(sys.argv) < 2:
    # Display usage instructions and examples if no input file is provided
    print(
        dedent("""
        Usage: llmparse.py <input.txt>
        Provide a text file where the contents are separated by paragraphs.
        Example:
        There is an airship today in the sky.

        Most cars on the road are currently red.
        
        Windows XP was a good operating system back then.
        
        Tools:
        llmparse.py <input.json> --to-history
        Converts an exported chat from OpenWeb UI to python Ollama history file.
        
        llmparse.py <input.json> --to-modelfile
        Converts a file containing history data to a modelfile (Only messages).
        
        llmparse.py <input.txt> --question-generator
        Runs a text file of paragraphs through the question-generator.
        
        llmparse.py <input.json> --benchmark
        Runs the benchmark on a python Ollama history file.
        """).strip()
    )

    # Exit the program with a non-zero status code to indicate an error
    sys.exit(1)

# Get the input file from the command-line arguments
inputFile = sys.argv[1]

# Check if the input file exists
if not os.path.isfile(inputFile):
    print(f"Error: File '{inputFile}' does not exist.")
    # Exit the program with a non-zero status code to indicate an error
    sys.exit(1)

# Function to get the context number from a history file for manual benchmarking
def get_ctx_num_from_history(file):
    """
    Calculate the context number based on the total character count in the history file.

    Args:
        file (str): The path to the history file.

    Returns:
        int: The calculated context number, or False if an error occurs.
    """

    try:
        # Open the file and read its contents
        with open(file, "r") as f:
            fileDataStr = f.read()
        
        # Parse the JSON data from the file
        history = json.loads(fileDataStr)

        # Initialize a counter for the total character count
        cCount = 0

        # Iterate over each message in the history and add its content length to the counter
        for message in history:
            cCount += len(message["content"])
        
        # Calculate and return the context number based on the total character count
        return token_block_size(cCount)
    except Exception as e:
        # Print an error message if any exception occurs during file parsing
        print(f"There was an error parsing file '{file}': {e}")
        return False

# Function to get a prompt for the LLM based on the specified task
def get_llm_prompt(which):
    # Use pattern matching to determine the correct prompt format
    match which:
        case 'question_generator':
            # Return a prompt for generating questions and answers from a given text
            return dedent("""
            Generate questions and answers from the given text starting with <text> and ending with </text>.
            The question must contain the context of the paragraph.

            Here is an example of how the output should look like:

            <example>
            What color is the sky?
            The sky is blue.

            How many fingers does a hand have?
            5.
            </example>

            Do not summarize or explain.

            <text>
            {input_text}
            </text>
            """).strip()
        case 'question_validator':
            # Return a prompt for validating the accuracy of generated answers
            return dedent("""
            Between <START> and <END> you will find 3 lines.
            The first line is a question.
            The second line is the correct answer.
            The third line is an answer generated by a computer.
            
            Your task is to determine if the third line contains the answer provided in the second line in context to the question.
            If the third line contains information that isn't in the second line or in the question, it fails and you respond with 1.
            Don't use tags in your responses, only 0 or 1.
            If the meaning is the same, you reply with 0, if it's not, you reply with 1.

            <START>
            {question}
            {answer}
            {response}
            <END>
            """).strip()

"""
Reads the entire contents of a file into a string.

Args:
    file (str, optional): The path to the file to read. Defaults to the value of the global 'inputFile' variable.

Returns:
    str: The contents of the file as a single string.  Returns an empty string if the file cannot be opened.
"""
def obtain_raw_data(file=inputFile):
    with open(file, "r") as f:  # Use 'with' statement for automatic file closing
        rawData = f.read()
    return rawData

"""
Generates question/answer pairs from a text file using an LLM and saves them
to a JSON file.

Args:
    save_as (str, optional): The name of the JSON file to save the chat history to.
                                Defaults to "chat_history.json".
    use (str, optional): The path to the input text file. Defaults to inputFile
                            (assumed to be a global variable or parameter).

Returns:
    int: An estimated ctx_num based on the number of characters in the generated question/answer pairs.
"""
def question_generator(save_as="chat_history.json", use=inputFile):
    # Retrieve the prompt for the question generator.
    prompt = get_llm_prompt(which="question_generator")

    # Read the raw data from the input file.
    rawData = obtain_raw_data(file=use)

    # Split the raw data into paragraphs based on double newlines.
    rawDataParagraphs = rawData.split("\n\n")

    # Get the amount of paragraphs
    j = len(rawDataParagraphs)

    # Initialize a set to keep track of seen questions to avoid duplicates.
    seenQuestions = set()

    # Initialize a list to store the chat history (question/answer pairs).
    chatHistory = []

    # Initialize a counter for the total number of characters.
    characterCount = 0

    # Iterate through each paragraph
    for i in tqdm(range(j)):
        # Iterate through different temperature settings for the LLM
        for k in range(0, 11):
            # Generate a response from the LLM for the current paragraph and temperature
            response = (
                cleanup_question_generator(
                    text=run_llm(
                        prompt.format(input_text=rawDataParagraphs[i]), temperature=k / 10
                    )
                )
            )

            # Check if the LLM generated a valid response
            if not response == False:
                # Split the response into question-answer pairs
                questionPairs = [response[i:i+2] for i in range(0, len(response), 2)]

                # Iterate through the question-answer pairs
                for questionPair in questionPairs:
                    # Check for duplicate questions, empty questions/answers, and valid question format
                    if questionPair[0].strip() not in seenQuestions\
                                               and questionPair[0].strip() != ''\
                                               and questionPair[1].strip() != ''\
                                               and questionPair[0].strip().endswith("?"):
                        
                        # Create a question dictionary
                        question = {
                            'role': 'user',
                            'content': questionPair[0].strip()
                        }

                        # Create an answer dictionary
                        answer = {
                            'role': 'assistant',
                            'content': questionPair[1].strip()
                        }

                        # Add the question and answer to the chat history
                        chatHistory.append(question)
                        chatHistory.append(answer)

                        # Update the character count
                        characterCount += len(question["content"])
                        characterCount += len(answer["content"])

                        # Add the question to the set of seen questions
                        seenQuestions.add(questionPair[0])

    # Store the chat history in a JSON file
    with open(save_as, 'w') as f:
        json.dump(chatHistory, f, indent=4)

    # Return a value for num_ctx based on the character count.
    return token_block_size(characterCount)


def cleanup_question_generator(text):
    """
    Cleans the text generated by the LLM, removing tags and filtering unwanted content.

    This function removes specific tags used to delineate questions and answers
    from the LLM's output. It also filters out placeholder or irrelevant text
    and validates that the cleaned text contains an even number of lines,
    indicating a valid question-answer pairing.

    Args:
        text (str): The raw text output from the LLM.

    Returns:
        list: A list of strings representing the cleaned question-answer pairs,
              or False if the cleaning process fails.  Each element in the list
              is either a question or an answer.
    """

    # Replace tags with newlines to separate questions and answers
    cleaned = (text.replace("<question>", "\n")
                .replace("</question>", "\n")
                .replace("<answer>", "\n")
                .replace("</answer>", "\n")
                .replace("<text>", "\n")
                .replace("</text>", "\n")
                .replace("\n\n", "\n") # Remove redundant newlines
                .strip() # Remove leading/trailing whitespace
    )

    # Define a list of texts to filter out (placeholder content)
    filtered = ["this section", "this text", "this paragraph", "the section", "the text", "the paragraph"]

    # Check if any filtered texts are present in the cleaned text
    for text in filtered:
        if text in cleaned:
            return False # Return False if unwanted text is found
    
    # Check if the cleaned text starts with a tag
    if text.startswith("<"):
        return False # Return False if a tag remains
    
    # Check if the cleaned text ends with a tag
    if text.endswith(">"):
        return False # Return False if a tag remains
    
    # Split the cleaned text into lines
    responseData = cleaned.split("\n")

    # Check if the number of lines is even (indicates valid question-answer pairs)
    if (len(responseData) % 2 != 0):
        return False # Return False if the number of lines is odd
    
    # Return the list of cleaned question-answer pairs
    return responseData


def token_block_size(characterCount):
    """
    Calculates the optimal token block size based on the character count.

    This function determines a suitable token block size for processing text,
    aiming to balance memory usage and processing efficiency. The calculation
    considers the character count to estimate the number of tokens and adjusts
    the target block size accordingly. It ensures that the final block size is a
    power of 2.

    Args:
        characterCount (int): The number of characters in the input text.

    Returns:
        int: The calculated token block size, which is a power of 2.
    """
    # Estimate the number of tokens (assuming roughly 4 characters per token)
    estimated_tokens = characterCount / 4

    # Determine the target block size based on the estimated number of tokens
    if estimated_tokens - 2048 < 4096:
        target = 4096
    elif estimated_tokens < 8096:
        target = estimated_tokens + 4096
    else:
        target = estimated_tokens + 2048
    
    # Round up to the nearest power of 2
    t = 4096 # Start with a reasonable power of 2
    while t < target:
        t *= 2
    
    return t

def extract_from_open_webui_export(inputFile, silent=False):
    """
    Extracts messages from an Open WebUI conversation export file.

    This function reads a JSON file exported from Open WebUI, which contains a
    conversation history. It parses the JSON data and extracts the messages,
    formatting them into a list of dictionaries, where each dictionary
    represents a message with 'role' and 'content' keys.

    Args:
        inputFile (str): The path to the Open WebUI export file (JSON).
        silent (bool): If True, suppresses error messages.  Defaults to False.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              message with 'role' and 'content'. Returns False if an error
              occurs during file processing.
    """

    messages = []
    try:
        with open(inputFile, "r") as f:  # Use "with" statement for automatic file closing
            exportDataStr = f.read()
            exportData = json.loads(exportDataStr)

            # Access the chat messages directly from the export data
            messages_list = exportData[0]["chat"]["messages"]

            # Populate the messages array with messages from the export data
            for message in messages_list:
                obj = {
                    'role': message["role"],
                    'content': message["content"]
                }
                messages.append(obj)
    except Exception as e:
        if not silent:
            print(f"There was an error converting the file: \n{e}")
        return False
    return messages

def run_llm_benchmark(question, answer, model, num_ctx):
    """
    Runs a benchmark test against a given LLM model.

    This function sends a question to the specified LLM, compares the response 
    to a known correct answer, and returns a boolean indicating whether the test passed.

    Args:
        question (str): The question to ask the LLM.
        answer (str): The correct answer to the question.
        model (str): The name of the LLM model to use (e.g., "llama2").
        num_ctx (int): The number of context tokens to use for the LLM.

    Returns:
        bool: True if the LLM's response is considered correct, False otherwise.
    """

    # Load chat history from a JSON file
    messages = [{
        'role': 'system',
        'content': 'You are a question and answer chatbot, ready to answer only questions based on the questions you already answered or related. Your response can\'t exceed 512 characters.'
    }] + json.loads(obtain_raw_data("chat_history.json"))

    # Append the user's question to the messages
    messages.append({
        'role': 'user',
        'content': question
    })

    # Send the messages to the LLM and get the response
    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            'num_keep': -1,
            'num_ctx': num_ctx
        }
    )
    responseContent = response.message.content

    # Print testing information
    print(f"Testing: {model}")
    result = run_llm(get_llm_prompt(which="question_validator").format(question=question, answer=answer, response=responseContent))
    print("Question: " + question)
    print("Answer (Correct): " + answer)
    print(f"Answer {model}: " + responseContent + "\n")

    # Determine if the test passed based on the result from run_llm
    passed = False
    if '0' in result:
        passed = True
    elif '1' in result:
        passed = False
    else:
        print(f"Failed: Couldn't interpret the models result: {result}")

    return passed

def history_benchmark(num_ctx=False, inputFile="chat_history.json"):
    """
    Runs a benchmark test against multiple LLMs using a chat history file.
    This function loads a chat history from a JSON file, selects random question-answer pairs,
    and tests each LLM's ability to correctly answer the questions.  It handles potential 
    errors during the benchmark process and returns the name of the best performing LLM.
    Args:
        num_ctx (int or bool, optional): The number of context tokens to use for the LLMs. 
                                         If False, it retrieves the context number from the history file.
                                         Defaults to False.
        inputFile (str, optional): The path to the JSON file containing the chat history. 
                                  Defaults to "chat_history.json".
    Returns:
        str: The name of the LLM with the highest number of passed tests.
    """
    
    # If num_ctx is not provided, retrieve it from the history file
    if num_ctx == False:
        num_ctx = get_ctx_num_from_history(file=inputFile)

    # Load message history from the input file
    msgHistory = json.loads(obtain_raw_data(inputFile))

    # Limit benchmark to 10 or the length of the history
    if len(msgHistory) < 10:
        benchmarkAmount = len(msgHistory)
    else:
        benchmarkAmount = 10

    # Creates an empty set which will be used to make sure all questions are unique
    usedIndices = set()

    # Creates an empty array to keep the selected question and answer pairs.
    benchmarkQuestions = []

    # Select random question-answer pairs from the message history
    while len(benchmarkQuestions) < benchmarkAmount:
        randomIndex = random.randint(0, len(msgHistory) - 1)
        if randomIndex not in usedIndices:
            if msgHistory[randomIndex]['role'] == 'assistant':
                # Append the previous user question and current assistant answer
                benchmarkQuestions.append(msgHistory[randomIndex - 1])
                benchmarkQuestions.append(msgHistory[randomIndex])
                usedIndices.add(randomIndex)
                usedIndices.add(randomIndex - 1)
            elif msgHistory[randomIndex]['role'] == 'user':
                # Append the current user question and next assistant answer
                benchmarkQuestions.append(msgHistory[randomIndex])
                benchmarkQuestions.append(msgHistory[randomIndex + 1])
                usedIndices.add(randomIndex)
                usedIndices.add(randomIndex + 1)

    # Initialize a dictionary to store the results of each LLM
    results = {}

    print(f"Running benchmarks with num_ctx={num_ctx}")

    # Run the benchmark for each selected question-answer pair and LLM
    for i in range(benchmarkAmount):
        for LLM in LLMs:
            if (benchmarkQuestions[i]['role'] == 'assistant'):
                # Use the previous user question as input
                question = benchmarkQuestions[i - 1]['content']
                answer = benchmarkQuestions[i]['content']
                
            elif (benchmarkQuestions[i]['role'] == 'user'):
                # Use the current user question as input and next assistant answer as expected output
                question = benchmarkQuestions[i]['content']
                answer = benchmarkQuestions[i + 1]['content']

            finished = False
            while finished == False:
                try:
                    passed = run_llm_benchmark(question=question, answer=answer, model=LLM, num_ctx=num_ctx)
                    finished = True
                except Exception as e:
                    print(f"An error occurred while trying to run the benchmark: {e}\nTrying again in 3 seconds.")
                    sleep(3)
                    print(f"Trying again to check {LLM}: Question: {question}\nAnswer: {answer}\n")
                
            # Update the results dictionary based on whether the LLM passed or failed
            if (passed):
                print(LLM + " passed the test.\n")
                if not LLM in results:
                    results[LLM] = 1
                else:
                    results[LLM] += 1
            else:
                print(LLM + " did not pass the test.\n")

    # Sort and print the results
    sortedResults = sorted(results.items(),key=operator.itemgetter(1),reverse=True)
    for key, value in sortedResults:
        print(f"{key}: {value}")
    
    # Return the name of the LLM with the highest number of passed tests
    return sortedResults[0][0]

# Runs an LLM model on a prompt with given temperature
def run_llm(prompt, model = "mistral:7b-instruct", temperature=0, num_ctx=4096):
    # The "model" parameter specifies which model to use. Default is "mistral:7b-instruct".
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx
        },
        raw=True
    )
    # The "response" variable holds the output of the LLM model.
    return response["response"]

# Check if the correct number of command line arguments are provided
if (len(sys.argv) == 3):
    # Determine which operation to perform based on the second command line argument

    # Convert Open WebUI export file to history format.
    if (sys.argv[2] == "--to-history"):
        # Construct the target file name by appending "-messages.json" to the input file stem
        targetFile = Path(inputFile).stem+"-messages.json"

        # Extract messages from the Open WebUI export file
        messages = extract_from_open_webui_export(inputFile)

        # Exit if no messages were extracted
        if not messages:
            exit(1)

        # Write the extracted messages to the target file in JSON format
        with open(targetFile, "w") as f:
            f.write(json.dumps(messages, indent=4))

        print("Done writing " + targetFile)
        exit(0)

    # Convert Open WebUI export file to model file format.
    if (sys.argv[2] == "--to-modelfile"):
        # Construct the target file name by appending "-messages.txt" to the input file stem
        targetFile = Path(inputFile).stem+"-messages.txt"

        # Remove the extra "-messages" suffix if present
        if targetFile.endswith("-messages.txt"):
            targetFile = targetFile.replace("-messages", "")
        
        # Extract messages from the Open WebUI export file in silent mode
        messages = extract_from_open_webui_export(inputFile, silent=True)

        # If extraction fails, attempt to read Ollama Python format
        if (messages == False):
            try:
                with open(inputFile, "r") as f:
                    messagesStr = f.read()
                messages = json.loads(messagesStr)

                # Check if the first message contains a 'role' key
                if not 'role' in messages[0]:
                    print("Could not find role in first message, input data is invalid.")
                    exit(1)
            except Exception as e:
                print(f"An error occured trying to parse the file:\n{e}")
                exit(1)

        # Exit if no messages were extracted
        if not messages:
            exit(1)
        
        # Construct the model file text by concatenating message role and content
        text = ""
        try:
            for msg in messages:
                text += "MESSAGE" + " " + msg["role"] + " " + msg["content"] + "\n"
        except Exception as e:
            print(f"An invalid object was found in the dataset:\n{e}")
            exit(1)
        
        # Write the model file text to the target file
        with open(targetFile, "w") as f:
            f.write(text)
        
        print(f"Done writing {targetFile}")
        exit(0)
    
    # Generate questions and save them to a JSON file.
    if (sys.argv[2] == "--question-generator"):
        # Construct the target file name by appending "-questions.json" to the input file stem
        targetFile = Path(inputFile).stem+"-questions.json"

        # Generate questions and save them to the target file
        tokens = question_generator(save_as=targetFile)

        print(f"Questions saved here: {targetFile}\nRecommended token size: {tokens}")
        exit(0)

    # Run a benchmark on the input file.
    if (sys.argv[2] == "--benchmark"):
        history_benchmark(inputFile=inputFile)
        exit(0)

    print("Too many parameters passed.")
    exit(1)

# Print an example system message
print("Example system message: You are a chat bot assistant ready to help using the data provided.")

# Prompt the user to provide a system message, defaulting to a standard Q&A message if left blank
system_message = input("Provide a system message [Leave blank for a standard Q&A]: ")

# Set the system message to the default Q&A message if the user didn't enter anything
if system_message == '':
    system_message = "You are a question and answer chatbot, ready to answer only questions based on the questions you already answered or related."

# Print an empty line for readability
print("\n")

# Print an example temperature input
print("Example input: 1")

# Prompt the user to provide a temperature value, defaulting to 0.8 if left blank
temperature = input("Please input a temperature [Leave blank for 0.8]: ")

# Set the temperature to the default value (0.8) if the user didn't enter anything
if temperature == '':
    temperature = 0.8
else:
    temperature = float(temperature)

# Function to generate a Modelfile based on the provided chat history and system message
def modelfile_generator(system="You are a question and answer chatbot, ready to answer only questions based on the questions you already answered or related.",
                        num_ctx=4096):

    # Load the chat history from the JSON file
    msgHistory = json.loads(obtain_raw_data("chat_history.json"))

    # Initialize an empty string to store the converted message history
    msgHistoryConverted = ""

    # Determine which model to use based on a benchmark test
    using = history_benchmark(num_ctx=num_ctx)

    # Iterate over each message in the chat history and convert it to the Modelfile format
    for msg in msgHistory:
        # Append each message to the converted message history string
        msgHistoryConverted += "MESSAGE " + msg["role"] + " " + msg["content"] + "\n"
    
    # Create the template for the Modelfile
    template = dedent("""
        FROM {using}
        PARAMETER num_keep -1
        PARAMETER num_ctx {num_ctx}
        PARAMETER temperature {temperature}
        SYSTEM {system}
        {msgHistoryConverted}
    """).strip().format(using=using, num_ctx=num_ctx, temperature=temperature, system=system, msgHistoryConverted=msgHistoryConverted)

    # Open the Modelfile in write mode and write the template to it
    with open('Modelfile', 'w') as f:
        # Write the template to the file
        f.write(template)

# Start the question generator, which will use the inputFile by default.
num_ctx = question_generator()

# Start the Modelfile generator, which will start the benchmark and use it's output.
modelfile_generator(system=system_message, num_ctx=num_ctx)
