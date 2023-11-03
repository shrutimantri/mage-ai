from typing import IO, Any, Dict, List, Union

from pandas import DataFrame, Series
from pinotdb import connect
from pinotdb.db import Connection

from mage_ai.io.base import QUERY_ROW_LIMIT, ExportWritePolicy
from mage_ai.io.config import BaseConfigLoader, ConfigKey
from mage_ai.io.sql import BaseSQL

import re

WRITE_NOT_SUPPORTED_EXCEPTION = Exception('write operations are not supported.')


class ConnectionWrapper(Connection):
    def cursor(self):
        cursor = connect(
            host=self._kwargs.get('host'), 
            port=self._kwargs.get('port'),
            username=self._kwargs.get('username'),
            password=self._kwargs.get('password'),
            path=self._kwargs.get('path'),
            scheme=self._kwargs.get('scheme'),
        ).cursor()
        return cursor

    def rollback(self):
        pass


class Pinot(BaseSQL):
    """
    Handles data transfer between a Pinot data warehouse and the Mage app.
    """

    def __init__(
            self,
            host: str,
            port: int,
            username: Union[str, None] = None,
            password: Union[str, None] = None,
            path: str = '/query/sql',
            scheme: str = 'http',
            **kwargs,
    ) -> None:
        """
        Initializes the data loader.

        Args:
            host (str): The host of the pinot controller to connect to.
            port (int): Port on which the pinot controller is running.
            username (str): The user with which to connect to the server with.
            password (str): The login password for the user.
            path (str): Path to pinot sql api.
            scheme (str): The scheme identifies the protocol to be used http or https.
            **kwargs: Additional settings for creating SQLAlchemy engine and connection
        """
        self._ctx = None
        super().__init__(
            host=host,
            port=port,
            username=username,
            password=password,
            path=path,
            scheme=scheme,
            **kwargs,
        )

    @classmethod
    def with_config(cls, config: BaseConfigLoader) -> 'Pinot':
        return cls(
            host=config[ConfigKey.PINOT_HOST],
            port=config[ConfigKey.PINOT_PORT],
            username=config[ConfigKey.PINOT_USER],
            password=config[ConfigKey.PINOT_PASSWORD],
            path=config[ConfigKey.PINOT_PATH],
            scheme=config[ConfigKey.PINOT_SCHEME],
        )

    def open(self) -> None:
        with self.printer.print_msg('Opening connection to Pinot warehouse'):
            connect_kwargs = dict(
                host=self.settings['host'],
                port=self.settings['port'],
                username=self.settings['username'],
                password=self.settings['password'],
                path=self.settings['path'],
                scheme=self.settings['scheme'],
            )
            print(connect_kwargs)
            self._ctx = ConnectionWrapper(**connect_kwargs)

    def table_exists(self, schema_name: str, table_name: str) -> bool:
        # Not used
        with self.conn.cursor() as cur:
            cur.execute(
                f'SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = \'{table_name}\''
            )
        return bool(cur.rowcount)

    def execute(self, query_string: str, **query_vars) -> None:
        super().execute(query_string, **query_vars)

    def execute_queries(self,
                        queries: List[str],
                        query_variables: List[Dict] = None,
                        commit: bool = False,
                        fetch_query_at_indexes: List[bool] = None) -> List:
        return super().execute_queries(queries, query_variables, commit, fetch_query_at_indexes)

    def fetch_query(self, cursor, query: str) -> Any:
        return super().fetch_query(cursor, query)

    def load(self,
             query_string: str,
             limit: int = QUERY_ROW_LIMIT,
             display_query: Union[str, None] = None,
             verbose: bool = True,
             **kwargs) -> DataFrame:
        return super().load(query_string, limit, display_query, verbose, **kwargs)

    def clean(self, column: Series, dtype: str) -> Series:
        return super().clean(column, dtype)

    def close(self) -> None:
        super().close()

    @property
    def conn(self) -> Any:
        return super().conn

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        """
        Close the underlying connection to the SQL data source if open. Else will do nothing.
        """
        if '_ctx' in self.__dict__:
            #self._ctx.close()
            del self._ctx

    def get_type(self, column: Series, dtype: str) -> str:
        if dtype in (
            PandasTypes.MIXED,
            PandasTypes.UNKNOWN_ARRAY,
            PandasTypes.COMPLEX,
        ):
            series = column[column.notnull()]
            values = series.values

            column_type = None

            if len(values) >= 1:
                column_type = 'JSONB'

                values_not_empty_list = [v for v in values if type(v) is not list or v]
                if not values_not_empty_list:
                    # All values are empty list
                    return column_type
                value = values_not_empty_list[0]
                if isinstance(value, list):
                    if len(value) >= 1:
                        item = value[0]
                        if type(item) is dict:
                            column_type = 'JSONB'
                        else:
                            item_series = pd.Series(data=item)
                            item_dtype = item_series.dtype
                            if PandasTypes.OBJECT != item_dtype:
                                item_type = self.get_type(item_series, item_dtype)
                                column_type = f'{item_type}[]'
                            else:
                                column_type = 'text[]'
                    else:
                        column_type = 'text[]'

            if column_type:
                return column_type

            raise BadConversionError(
                f'Cannot convert column \'{column.name}\' with data type \'{dtype}\' to '
                'a PostgreSQL datatype.'
            )
        elif dtype in (PandasTypes.DATETIME, PandasTypes.DATETIME64):
            try:
                if column.dt.tz:
                    return 'timestamptz'
            except AttributeError:
                pass
            return 'timestamp'
        elif dtype == PandasTypes.TIME:
            try:
                if column.dt.tz:
                    return 'timetz'
            except AttributeError:
                pass
            return 'time'
        elif dtype == PandasTypes.DATE:
            return 'date'
        elif dtype == PandasTypes.STRING:
            return 'text'
        elif dtype == PandasTypes.CATEGORICAL:
            return 'text'
        elif dtype == PandasTypes.BYTES:
            return 'bytea'
        elif dtype in (PandasTypes.FLOATING, PandasTypes.DECIMAL, PandasTypes.MIXED_INTEGER_FLOAT):
            return 'double precision'
        elif dtype == PandasTypes.INTEGER or dtype == PandasTypes.INT64:
            max_int, min_int = column.max(), column.min()
            if np.int16(max_int) == max_int and np.int16(min_int) == min_int:
                return 'smallint'
            elif np.int32(max_int) == max_int and np.int32(min_int) == min_int:
                return 'integer'
            else:
                return 'bigint'
        elif dtype == PandasTypes.BOOLEAN:
            return 'boolean'
        elif dtype in (PandasTypes.TIMEDELTA, PandasTypes.TIMEDELTA64, PandasTypes.PERIOD):
            return 'bigint'
        elif dtype == PandasTypes.EMPTY:
            return 'text'
        elif PandasTypes.OBJECT == dtype:
            return 'JSONB'
        else:
            print(f'Invalid datatype provided: {dtype}')

        return 'text'

    def upload_dataframe(self,
                         cursor,
                         df: DataFrame,
                         db_dtypes: List[str],
                         dtypes: List[str],
                         full_table_name: str,
                         buffer: Union[IO, None] = None,
                         **kwargs) -> None:
        raise WRITE_NOT_SUPPORTED_EXCEPTION

    def export(self,
               df: DataFrame,
               schema_name: str,
               table_name: str,
               if_exists: ExportWritePolicy = ExportWritePolicy.REPLACE,
               index: bool = False, verbose: bool = True,
               query_string: Union[str, None] = None,
               drop_table_on_replace: bool = False,
               cascade_on_drop: bool = False,
               allow_reserved_words: bool = False,
               unique_conflict_method: str = None,
               unique_constraints: List[str] = None) -> None:
        raise WRITE_NOT_SUPPORTED_EXCEPTION

    def _enforce_limit(self, query: str, limit: int = QUERY_ROW_LIMIT) -> str:
        query = query.strip(';')
        if re.search('limit', query, re.IGNORECASE):
            return f"""
{query}
"""
        else:
            return f""" 
{query} limit {limit}
"""