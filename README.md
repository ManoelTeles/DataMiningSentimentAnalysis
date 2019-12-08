Criar Ambiente Virtual

    virtualenv -p python3 env

Instalar pacotes

    - Entrar no ambiente

        source env/bin/activate

    - Depois de estar no Ambiente
    
        pip install -r requirements.txt

É necessario a instalação do seguinte pacote com as bibliotecas adicionais
    
    python -m textblob.download_corpora

run in python3 

    import nltk
    nltk.download('stopwords')

