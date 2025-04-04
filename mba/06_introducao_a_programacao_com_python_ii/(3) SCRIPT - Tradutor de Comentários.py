# -*- coding: utf-8 -*-

# Introdução à Programação com Python
# MBA em Data Science e Analytics USP ESALQ

#%% Instalando o pacote necessário

!pip install googletrans==4.0.0-rc1

#%% Importando os pacotes

from googletrans import Translator
import time

#%% Indicar o texto que deverá ser traduzido

# Nota: manter as aspas (""") no início e fim (o texto deve ficar entre elas)

translator = Translator()

code = """

#%% Numérico

# Na função a seguir, note que é comum atribuir apelidos aos pacotes "pd"
# A partir de agora, sempre que usarmos o pandas será com o nome "pd"
# Na sequência, indicamos a função "Series" que está no pacote "pd"

numeros = pd.Series([10,20,30,40,50,60,70,80])

print(numeros)

"""
#%% Definindo a função (se necessário, alterar o idioma destino na linha 36)

def translate_comment(comment, translator, retries=5, delay=2):
    for attempt in range(retries):
        try:
            translated_comment = translator.translate(comment,
                                                      src='pt', dest='en').text
            return translated_comment
        except Exception as e:
            print(f"Erro ao traduzir: {e}. Tentativa {attempt + 1} de {retries}.")
            time.sleep(delay)
    return comment

def translate_comments(code, translator):
    lines = code.split('\n')
    translated_code = []
    for line in lines:
        if '#' in line:
            code_part, comment_part = line.split('#', 1)
            translated_comment = translate_comment(comment_part.strip(), translator)
            translated_code.append(f"{code_part.strip()} # {translated_comment}")
        else:
            translated_code.append(line)
    return '\n'.join(translated_code)

#%% Gerando a tradução

translated_code = translate_comments(code, translator)
print(translated_code)

#%% Fim!