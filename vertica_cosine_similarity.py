from vertica_python import connect

from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

import ast

def safe_eval(x):
    vec = x + "]"
    try:
        return np.array(ast.literal_eval(vec))[:60]
    except (SyntaxError, ValueError):
        return np.nan

connection = connect(
    host='bl1.cs.fsu.edu',
    port=5433,
    user='dbadmin',
    password='23222322',
    database='cankg'
)

similarity_array = []


sim_vec = safe_eval(connection.cursor().execute("select embeddings from emb_ckg e where e.word = 'drugs' limit 1").fetchall()[0][0])

cursor = connection.cursor()
query = "SELECT word from emb_ckg"
cursor.execute(query)
word_dataframe = np.array(cursor.fetchall()).flatten()

# for row in word_dataframe:
#     hash_table[row] = 0

count = int( connection.cursor().execute("select count(table_id) from emb_ckg").fetchall()[0][0] )

# print(count)

for i in range(0, count, 10000):
    batch = connection.cursor().execute(f"select embeddings from emb_ckg offset {i} limit 10000").fetchall()

    for j in range(10000):
        if j >= len(batch): break
        emb_vec = safe_eval(batch[j][0])
        similarity_array.append( 1 - cosine(emb_vec, sim_vec) )

    print(i)


word_dataframe = pd.DataFrame({"word": word_dataframe, "similarity": similarity_array})

# sorted_results = word_dataframe.sort(reverse = True, key = lambda x: hash_table[x])

word_dataframe = word_dataframe.sort_values(by='similarity', ascending=False)

# sorted_df.reset_index(drop=True, inplace=True)


word_dataframe.to_csv("vertica_similarity.csv", index=None)


cursor.close()
connection.close()