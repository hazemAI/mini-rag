from dotenv import dotenv_values
config = dotenv_values(".env")

port = 5555
max_tasks = 10000
auto_refersh = True


basic_auth = [f'admin:{config["CELERY_FLOWER_PASSWORD"]}']
