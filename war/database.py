from tempfile import NamedTemporaryFile
import logging
import os
import pickle
import zlib


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
            logger.debug('\033[38;5;240mInitialized empty database %s\033[0m', self.db_path)

    def _write_object(self, oid, content):
        object_path = os.path.join(self.db_path, oid[:2], oid[2:])
        dirname = os.path.dirname(object_path)

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        logger = logging.getLogger('war.database')
        tmp = NamedTemporaryFile(dir=dirname, prefix='obj_', delete=False)
        try:
            logger.debug('\033[38;5;240mWriting compressed object in %s\033[0m', tmp.name)
            compressed = zlib.compress(content, level=9)
            tmp.write(compressed)
            tmp.close()
            logger.debug('\033[38;5;240mCompression: %d of %d (%.2f%% of original size)\033[0m',
                         len(compressed), len(content),
                         100 * len(compressed) / len(content))
            logger.debug('\033[38;5;240mRenaming compressed object to %s\033[0m', object_path)
            os.rename(tmp.name, object_path)
        finally:
            if os.path.exists(tmp.name):
                os.delete(tmp.name)

    def _object_path(self, id):
        return os.path.join(self.db_path, id[:2], id[2:])

    def iterate(self):
        logger = logging.getLogger('war.database')
        files = []
        for root, dirs, files in os.walk(self.db_path):
            for file in files:
                id = os.path.basename(root) + file
                yield (id, self.load(id))

    def find(self, id):
        return os.path.exists(self._object_path(id))

    def load(self, id):
        if not self.find(id):
            return None
        logger = logging.getLogger('war.database')
        object_path = self._object_path(id)
        logger.debug('\033[38;5;240mLoading object %s\033[0m',
                     object_path)
        with open(object_path, 'rb') as file:
            decompressed = zlib.decompress(file.read())
            obj = pickle.loads(decompressed)
            logger.debug('\033[38;5;240mLoaded object %s', obj)
            return obj

    def store(self, id, object):
        logger = logging.getLogger('war.database')
        logger.debug('\033[38;5;240mStoring object %s in namespace %s\033[0m',
                     id, self.namespace)
        self._write_object(id, pickle.dumps(object))
