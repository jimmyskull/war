"""Database implementation."""
from tempfile import NamedTemporaryFile
import logging
import os
import pickle
import zlib

from war.cformat import ColorFormat as CF


DEFAULT_CHEF_DIR_ENVIRONMENT = '.war'


class Database:
    """
    Database controls object stored in the disk.

    The database is used to store results. In case of results, objects are
    named by the Task's SHA-1, not the object contents itself.

    By default, the database is located at `.war/`.
    """

    def __init__(self, namespace='default'):
        self.namespace = namespace
        self.db_path = os.path.join(DEFAULT_CHEF_DIR_ENVIRONMENT, namespace)
        self._init()

    def _init(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            logger = logging.getLogger('war.database')
            logger.debug(CF('Initialized empty database %s').dark_gray,
                         self.db_path)

    def _object_path(self, oid):
        # Return .war/namespace/00/00000..
        return os.path.join(self.db_path, oid[:2], oid[2:])

    def _write_object(self, oid, content):
        object_path = self._object_path(oid)
        dirname = os.path.dirname(object_path)

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        logger = logging.getLogger('war.database')
        tmp = NamedTemporaryFile(dir=dirname, prefix='obj_', delete=False)
        try:
            logger.debug(CF('Writing compressed object in %s').dark_gray,
                         tmp.name)
            compressed = zlib.compress(content, level=9)
            tmp.write(compressed)
            tmp.close()
            logger.debug(
                CF('Compression: %d of %d (%.2f%% of original size)').dark_gray,
                len(compressed), len(content),
                100 * len(compressed) / len(content))
            logger.debug(CF('Renaming compressed object to %s').dark_gray,
                         object_path)
            os.rename(tmp.name, object_path)
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

    def iterate(self):
        """Iterate over all objects in the database."""
        files = []
        for root, unused_dirs, files in os.walk(self.db_path):
            for file in files:
                if file.startswith('obj_'):
                    # Skip temporary files.
                    continue
                oid = os.path.basename(root) + file
                yield (oid, self.load(oid))

    def find(self, oid):
        """
        Find an object in the database.

        Returns
        -------
        bool
            Whether the object exists in the database.
        """
        return os.path.exists(self._object_path(oid))

    def load(self, oid):
        """
        Return an object from the database.

        Parameters
        ----------
        oid : str
            The object id, a SHA-1 hex digest.

        Returns
        -------
        object
            The object from the database. None when the object does
            not exist.
        """
        if not self.find(oid):
            return None
        object_path = self._object_path(oid)
        logger = logging.getLogger('war.database')
        logger.debug(CF('Loading object %s').dark_gray, object_path)
        with open(object_path, 'rb') as file:
            decompressed = zlib.decompress(file.read())
            obj = pickle.loads(decompressed)
            return obj

    def store(self, oid, obj):
        """
        Store an object in the database.

        Parameters
        ----------
        oid : str
            The object id, a SHA-1 hex digest.
        obj : object
            The object that will be pickled into the database.
        """
        logger = logging.getLogger('war.database')
        logger.debug(CF('Storing object %s/%s').dark_gray, self.namespace, oid)
        self._write_object(oid, pickle.dumps(obj))
