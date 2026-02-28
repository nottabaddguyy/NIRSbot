import logging
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import text_processing as tp

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

df = None
vectorizer = TfidfVectorizer()
tfidf_matrix = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение и инструкцию."""
    await update.message.reply_text(
        "👋 Здравствуйте! Я бот для поиска научных статей.\n"
        "Напишите интересующую вас тему, и я найду наиболее релевантные публикации.\n"
        "Например: методы NLP")

async def search_articles(query: str) -> list:
    global df, vectorizer, tfidf_matrix

    processed_query = tp.preprocess(query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.1: 
            row = df.iloc[idx]
            results.append({
                'title': row['title'],
                'authors': row['authors'],
                'subject': row['subject'],
                'abstract': row['abstract'][:200] + '...',  
                'url': row['url'],
                'score': round(score, 3)})
    return results

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    await update.message.reply_text("Ищу подходящие статьи...")

    results = await search_articles(query)
    if not results:
        await update.message.reply_text("К сожалению, ничего не найдено. Попробуйте изменить запрос.")
        return

    response = f"Нашёл {len(results)} статей по вашему запросу:\n\n"
    for i, art in enumerate(results, 1):
        response += (
            f"{i}. *{art['title']}*\n"
            f"   Авторы: {art['authors']}\n"
            f"   Рубрика ВИНИТИ: {art['subject']}\n"
            f"   Аннотация: {art['abstract']}\n"
            f"   [Ссылка на статью]({art['url']})\n\n")

    await update.message.reply_text(response, parse_mode='Markdown')

def load_data():
    global df, vectorizer, tfidf_matrix

    df = pd.read_csv('data.csv', encoding='utf-8')
    # Предобработка всех аннотаций
    processed_abstracts = [tp.preprocess(text) for text in df['abstract']]
    # Обучаем векторный преобразователь
    tfidf_matrix = vectorizer.fit_transform(processed_abstracts)
    logger.info(f"Загружено {len(df)} статей, матрица размером {tfidf_matrix.shape}")

def main() -> None:
    
    TOKEN = 'Вставьте токен'

    load_data()

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
