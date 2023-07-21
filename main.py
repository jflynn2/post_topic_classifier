import config as creds
import pandas as pd
import openai
import os
from  itertools import chain
from collections import Counter
import json
import glob
import logging
from time import sleep
from csv import DictReader
import psycopg2
import ast
from datetime import date

openai.api_key = creds.OPENAI_KEY


def connect():

    # Set up a connection to the postgres server.
    conn_string = "host="+ creds.PGHOST +" port="+ "5432" +" dbname="+ creds.PGDATABASE +" user=" + creds.PGUSER \
                  +" password="+ creds.PGPASSWORD

    conn = psycopg2.connect(conn_string)
    print("Connected!")

    # Create a cursor object
    cursor = conn.cursor()

    return conn, cursor


def test_conn():
    con, cursor = connect()
    select = "select * from public.users_competitors"
    cursor.execute(select)
    test_data = cursor.fetchall()
    return test_data


def select_n_random_posts(n):
    con, cursor = connect()
    select = "SELECT * FROM cm_combine.cm_harvest_results TABLESAMPLE system_rows(" + str(n) + ");"
    cursor.execute(select)
    columns = [item.name for item in cursor.description]
    result = pd.DataFrame(cursor.fetchall(), columns=columns)

    cursor.close()

    return result


def select_post_recent_accounts():
    con, cursor = connect()

    select = """select post_id, clean_text from cm_combine.cm_harvest_results
        where account_id IN (SELECT account_id from cm_combine.cm_harvest_accounts
        where last_updated > now() - interval '1 hour')
        and post_timestamp > now() - interval '2 week';"""

    cursor.execute(select)
    columns = [item.name for item in cursor.description]
    result = pd.DataFrame(cursor.fetchall(), columns=columns)

    cursor.close()
    return result


def select_recent_posts(days):
    con, cursor = connect()
    select = """(SELECT post_id, clean_text FROM cm_combine.cm_harvest_results
                WHERE post_timestamp > now() - interval '""" + str(days) + """ day'
                            AND topics = '{}')
            UNION
            (SELECT post_id, clean_text FROM cm_combine.cm_harvest_results
            WHERE username IN (select username from cm_combine.cm_harvest_accounts 
                            where posts_downloaded <= 25
                            AND platform = 'instagram'
                            AND last_updated > now () - interval '4 hours')       
                            AND topics = '{}'
                            AND platform = 'instagram')"""
    cursor.execute(select)
    columns = [item.name for item in cursor.description]
    result = pd.DataFrame(cursor.fetchall(), columns=columns)

    cursor.close()
    return result


def get_most_recent_post_from_IDs(ids):
    con, cursor = connect()

    select = """SELECT post_id, post_timestamp FROM cm_combine.cm_harvest_results WHERE post_id IN """ \
             + str(tuple(ids)) + """ORDER  BY post_timestamp DESC"""
    cursor.execute(select)
    list_of_posts = cursor.fetchall()
    cursor.close()
    return list_of_posts


def get_topics_from_openai(post, model_name="text-davinci-003"):

    response = openai.Completion.create(
        engine=model_name,
        prompt=f"""Output the topics of the following text as a list of nouns in a json object. 
        Here are a list of 4 good examples.
        Example 1. "Another goal for the hosts. 4-1." the output is {{"topics": ["goal", "MUFC", "WATMUN"]}}.
        Example 2. "Heartbroken mother pays tribute after boy dies in farmyard accident with tractor" the output is 
        {{"topics": ["Heartbroken mother", "boy", "death", "farmyard accident", "tractor"]}}.
        Example 3. "A brave effort from Scariff was not enough to stop three in a row chasing Sixmilebridge from 
        maintaining their unbeaten run." the output is {{"topics": ["Scariff", "Sixmilebridge", "unbeaten run"]}}. 
        Example 4. For the text "This Father's Day let's celebrate dads all over the world, dads like Abdifatah in 
        Ethiopia. :partying_face:" the output is {{"topics": ["Father's Day", "dads", "Abdifatah", "Ethiopia"]}}.
        The text to analyse is here: {post}""",
        # prompt=f"""Output the topics of the following text as a list of nouns in a json object: {post}""",
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.2,
    )
    return response.choices[0]['text'].strip()


def get_topics_from_openai_chat(post, model_name="gpt-3.5-turbo"):
    print(post)
    sleep_counter = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": f"""Output the topics of the following text as a list of nouns in a json object.
                If the topic does not appear in the text, do not output it. Here are a list of 4 good examples.
                Example 1. "Another goal for the hosts. 4-1. #MUFC #WATMUN" the output is {{"topics": ["goal", "MUFC", "WATMUN"]}}.
                Example 2. "A taste of home soil. :house_with_garden::footprints: OscarPiastri" the output is
                {{"topics": ['home soil', 'OscarPiastri']}}.
                Example 3. "A brave effort from Scariff was not enough to stop three in a row chasing Sixmilebridge from
                maintaining their unbeaten run." the output is {{"topics": ["Scariff", "Sixmilebridge", "unbeaten run"]}}.
                Example 4. "This Father's Day let's celebrate dads all over the world, dads like Abdifatah in
                Ethiopia. :partying_face:" the output is {{"topics": ["Father's Day", "dads", "Abdifatah", "Ethiopia"]}}.
                The text to analyse is here:  {post}"""}
                ],
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.2,
            )
            topics = response.choices[0]['message']["content"]
            break

        except openai.OpenAIError as oaiError:
            logging.log(level=30, msg=oaiError)
            sleep(sleep_counter)
            sleep_counter = (sleep_counter ^ 2) + 1

    try:
        post_topics = json.loads(topics)
    except json.JSONDecodeError:
        post_topics = {"topics": [""]}

    print(post_topics['topics'])

    return post_topics['topics']


def pass_posts_to_openai(posts, model_name):
    if model_name =='gpt-3.5-turbo':
        posts['topics'] = posts['clean_text'].apply(get_topics_from_openai_chat)

    else:
        print("Only gpt-3.5-turbo supported")
        exit()

    return posts


def find_dates_of_posts(file_path):
    all_posts = pd.read_csv(file_path)
    all_posts = all_posts[all_posts['post_id'].apply(lambda x: str(x).isdigit())]
    posts_ids = all_posts['post_id'].tolist()
    return get_most_recent_post_from_IDs(posts_ids)


def select_random_posts_classify(n=100, model_name='text-davinci-003'):
    posts = select_n_random_posts(n)['clean_text']

    topics_posts_df = pass_posts_to_openai(posts, model_name)

    os.makedirs('topic_outputs/davinci', exist_ok=True)
    topics_posts_df.to_csv('topic_outputs/davinci/topics_posts_df.csv', index=False)

    topics_counts = pd.Series(Counter(chain.from_iterable(topics_posts_df['topics'])))
    topics_counts.to_csv('topic_outputs/davinci/topics_counts.csv')


def insert_topics_in_db(topics_file):

    with open(topics_file) as f:
        with psycopg2.connect(
                'postgresql://cm_combine_apu:M4n4g3r1@database-1.clkphshltxwg.eu-west-1.rds.amazonaws.com:5432/connect_mor_prod') as db:
            cur = db.cursor()

            records = DictReader(f, delimiter=',')
            for i, record in enumerate(records):
                record['topics'] = [t.lower() for t in ast.literal_eval(record['topics'])]
                try:
                    cur.execute('UPDATE cm_combine.cm_harvest_results SET topics=%(topics)s where post_id=%(post_id)s;',
                                record)
                except Exception as e:
                    print(record)

                if i % 100 == 0:  # Commit every 100 rows
                    db.commit()
                    print(str(i))
            # Final Commit
            db.commit()


def combine_csvs(dir):

    master_df = pd.read_csv("topic_outputs/" + dynamic_date + "/topics_posts0.csv")
    for f in glob.glob('topic_outputs/' + dynamic_date + '/topics_posts*.csv'):
        print(str(f))
        append_df = pd.read_csv(f)
        master_df = pd.concat([master_df, append_df], axis=0)

    master_df = master_df.reset_index(drop=True)
    master_df.to_csv('topic_outputs/' + dynamic_date + '/master_topics.csv')
    print(master_df.shape)


if __name__ == "__main__":
    today = date.today()
    dynamic_date = today.strftime("%b_%d_%Y")
    # posts = pd.read_csv("topic_outputs/davinci-turbo/n=1000/topics_posts_df.csv")['text'].iloc[750:752]
    # posts = select_random_posts_classify(n=100, model_name='gpt-3.5-turbo')
    posts = select_recent_posts(2)
    os.makedirs('topic_outputs/' + dynamic_date + '', exist_ok=True)
    posts.to_csv('topic_outputs/' + dynamic_date + '/save_posts.csv', index=False)

    start = 0
    end_increment = 100
    end = start + end_increment
    all_posts = pd.read_csv('topic_outputs/' + dynamic_date + '/save_posts.csv')
    print(all_posts.shape)
    length = len(all_posts)
    while end < length:
        posts = all_posts.iloc[start:end]
        topics_posts_df = pass_posts_to_openai(posts, "gpt-3.5-turbo")
        print(topics_posts_df.head())
        topics_posts_df.to_csv('topic_outputs/' + dynamic_date + '/topics_posts'+str(start)+'.csv', index=False)
        start = end
        end = end + end_increment
        print("=================" + str(end) + "==============")

    posts = all_posts.iloc[start:end]
    topics_posts_df = pass_posts_to_openai(posts, "gpt-3.5-turbo")
    topics_posts_df.to_csv('topic_outputs/' + dynamic_date + '/topics_posts' + str(start) + '.csv', index=False)
    #
    combine_csvs('topic_outputs/' + dynamic_date + '')
    insert_topics_in_db('topic_outputs/' + dynamic_date + '/master_topics.csv')


    # # get topics for recent posts
    # posts = select_post_recent_accounts()
    # topics_posts_df = pass_posts_to_openai(posts, "gpt-3.5-turbo")
    # topics_posts_df.to_csv('topic_outputs/' + dynamic_date + '/topics_posts_recent_accounts.csv', index=False)

    # find_dates_of_posts('topic_outputs/' + dynamic_date + '/master_topics.csv')

    # topics_counts = pd.Series(Counter(chain.from_iterable(topics_posts_df['topics'])))
    # print(topics_counts.head())
    # topics_counts.to_csv('topic_outputs/davinci-turbo/topics_counts.csv')

    # master_df = pd.read_csv("topic_outputs/davinci-turbo/topics_posts0.csv")
    # for f in glob.glob('topic_outputs/davinci-turbo/topics_posts*.csv'):
    #     print('hello')
    #     append_df = pd.read_csv(f)
    #     master_df = pd.concat([master_df, append_df], axis=0)
    #
    # master_df = master_df.reset_index(drop=True)
    # master_df.to_csv('topic_outputs/davinci-turbo/master_topics.csv')
    # print(master_df.shape)
