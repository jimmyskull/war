from tempfile import NamedTemporaryFile
import logging
import os
import pickle
import zlib

from war.cformat import ColorFormat


DEFAULT_CHEF_DIR_ENVIRONMENT = '.war'


class Database:

    def __init__(self, namespace='default'):
        self.namespace = namespace
        self.db_path = os.path.join(DEFAULT_CHEF_DIR_ENVIRONMENT, namespace)
        self._init()

    def _init(self):
        logger = logging.getLogger('war.database')
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            logger.debug(
                ColorFormat('Initialized empty database %s').dark_gray,
                self.db_path)

    def _write_object(self, oid, content):
        object_path = os.path.join(self.db_path, oid[:2], oid[2:])
        dirname = os.path.dirname(object_path)

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        logger = logging.getLogger('war.database')
        tmp = NamedTemporaryFile(dir=dirname, prefix='obj_', delete=False)
        try:
            logger.debug(
                ColorFormat('Writing compressed object in %s').dark_gray,
                tmp.name)
            compressed = zlib.compress(content, level=9)
            tmp.write(compressed)
            tmp.close()
            logger.debug(
                ColorFormat(
                    'Compression: %d of %d (%.2f%% of original size)'
                ).dark_gray,
                len(compressed), len(content),
                100 * len(compressed) / len(content))
            logger.debug(
                ColorFormat('Renaming compressed object to %s').dark_gray,
                object_path)
            os.rename(tmp.name, object_path)
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

    def _object_path(self, oid):
        return os.path.join(self.db_path, oid[:2], oid[2:])

    def iterate(self):
        files = []
        for root, dirs, files in os.walk(self.db_path):
            for file in files:
                oid = os.path.basename(root) + file
                yield (oid, self.load(oid))

    def find(self, oid):
        return os.path.exists(self._object_path(oid))

    def load(self, oid):
        if not self.find(oid):
            return None
        logger = logging.getLogger('war.database')
        object_path = self._object_path(oid)
        logger.debug(ColorFormat('Loading object %s').dark_gray, object_path)
        with open(object_path, 'rb') as file:
            decompressed = zlib.decompress(file.read())
            obj = pickle.loads(decompressed)
            return obj

    def store(self, oid, obj):
        logger = logging.getLogger('war.database')
        logger.debug(ColorFormat('Storing object %s/%s').dark_gray,
                     self.namespace, oid)
        self._write_object(oid, pickle.dumps(obj))
