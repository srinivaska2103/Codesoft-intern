

def zack():
    print("Chatbot: Hi! I'm your friendly chatbot. Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ").lower()
        
    
        if "bye" in user_input:
            print("zack: Goodbye! Have a nice day 😊")
            break
        
      
        elif "hello" in user_input or "hi" in user_input:
            print("zack: Hello there! How can I help you today?")
        
       
        elif "how are you" in user_input:
            print("zack: I'm just a program, but I'm doing great! How about you?")
        
       
        elif "your name" in user_input:
            print("zack: I'm a simple Python chatbot created by you!")
        
    
        elif "weather" in user_input:
            print("zack: I can't check real-time weather yet, but I hope it's nice where you are!")
        
        
        else:
            print("zack: I'm not sure how to respond to that. Can you ask something else?")
            

# Run the chatbot
zack()
