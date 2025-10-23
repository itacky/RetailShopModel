from app.util.bigquery.storage import runQuery
from config import settings

QUERY_LOCATION = f'{settings.PROJECT_DIR}/app/services/bq_sql/queries'


class CustomerLifecycleGCP():
    def __init__(self):
        self.create_query_file = f'{QUERY_LOCATION}/customer_lifecycle_create.sql'

    def process(self):
        runQuery(self.create_query_file)