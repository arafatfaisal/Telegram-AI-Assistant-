from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import torchvision
torchvision.disable_beta_transforms_warning()


# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate response (Make it async)
async def generate_response(prompt):
    try:
        # Enhance the prompt to make the model act like a helpful assistant
        enhanced_prompt = f"You are a helpful assistant. Answer the user's questions directly.\nUser: {prompt}\nAssistant:"
        
        # Tokenize input text
        inputs = tokenizer(enhanced_prompt, return_tensors="pt")
        # Generate the response asynchronously
        outputs = await asyncio.to_thread(model.generate, inputs["input_ids"], max_length=200, num_return_sequences=1, )
        
        # Decode the generated tokens to a string
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return the response
        return response.split("Assistant:")[-1].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error while processing your request."

# Respond to the /start command
async def start(update: Update, context):
    print("Received /start command")
    await update.message.reply_text("Hello! I am your AI assistant. Ask me anything!")

# Handle user messages
async def handle_message(update: Update, context):
    user_message = update.message.text
    print(f"User: {user_message}")

    # Generate response using the LLM
    ai_response = await generate_response(user_message)
    print(f"AI: {ai_response}")

    # Send AI response back to the user
    await update.message.reply_text(ai_response)

# Main function
if __name__ == "__main__":
    print("Starting bot...")

    app = ApplicationBuilder().token("7501265647:AAGdHzQpyx82BCyUXVvdUFYZYN3IHGrUQko").build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Bot is running...")
    app.run_polling()
