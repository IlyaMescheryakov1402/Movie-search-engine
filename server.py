from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time

# Запускаем приложение
app = Flask(__name__, template_folder='.')

# Считывает сырые данные и строит инвертированный индекс
build_index()

# Применяем декоратор маршрутизации
@app.route('/', methods=['GET'])
def index():
    start_time = time()

    # Получаем запрос
    query = request.args.get('query')

    # Отлавливаем случай, когда в очереди ничего нет
    if query is None:
        query = ''
    
    # Получаем список релевантных документов
    documents = retrieve(query)

    # Считаем релевантность для каждого документа из списка
    scored = [(doc, score(query, doc)) for doc in documents]

    # Сортируем документы по релевантности
    scored = sorted(scored, key=lambda doc: -doc[1])

    # Делаем нужный формат для отображения
    results = [doc.format(query)+['%.2f' % scr] for doc, scr in scored]

    # Возвращаем функцию фласка для отображения результатов на форме
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Eliajah',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
