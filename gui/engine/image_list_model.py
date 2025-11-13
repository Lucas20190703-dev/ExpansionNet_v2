import os
from PySide2.QtCore import (QAbstractListModel, Qt, QModelIndex, QObject, Property,
                            QDateTime, Slot, QUrl, Signal)

from gui.engine.string_similarity import *

class ImageInfo(QObject):
    def __init__(self, file_path, captions=[], parent=None):
        super().__init__(parent)
        self._file_path = file_path
        self._file_name = os.path.basename(file_path)
        self._file_size = os.path.getsize(file_path)
        self._modified_time = QDateTime.fromSecsSinceEpoch(int(os.path.getmtime(file_path)))
        self._captions = captions

    @Property(str)
    def filePath(self):
        return QUrl.fromLocalFile(self._file_path).toString()

    @Property(str)
    def fileName(self):
        return self._file_name

    @Property(int)
    def fileSize(self):
        return self._file_size

    @Property(QDateTime)
    def modifiedTime(self):
        return self._modified_time

    @Property(list)
    def cations(self):
        return self._captions


class ImageListModel(QAbstractListModel):
    FilePathRole = Qt.UserRole + 1
    FileNameRole = Qt.UserRole + 2
    FileSizeRole = Qt.UserRole + 3
    ModifiedTimeRole = Qt.UserRole + 4
    CaptionsRole = Qt.UserRole + 5
    
    countChanged = Signal()
    canLoadMoreChanged = Signal()

    def __init__(self, image_paths=None, parent=None):
        super().__init__(parent)
        self._all_paths = image_paths or []
        self._all_paths_filtered = self._all_paths
        
        self._images = []
        self._batch_size = 50
        self._loaded = 0
        self._search_text = ""
        self._search_caption = ""
        self._search_caption_similarity = 0.5
        self._search_start_date = None #QDateTime.currentDateTime().date()
        self._search_end_date = None #QDateTime.currentDateTime().date()
        
        self._database = None
        
        if self._loaded < len(self._all_paths_filtered):
            self.canLoadMoreChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return len(self._images)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        image = self._images[index.row()]
        if role == self.FilePathRole:
            return image.filePath
        if role == self.FileNameRole:
            return image.fileName
        if role == self.FileSizeRole:
            return image.fileSize
        if role == self.ModifiedTimeRole:
            return image.modifiedTime
        if role == self.CaptionsRole:
            return image.captions
        return None

    def roleNames(self):
        return {
            self.FilePathRole: b'filePath',
            self.FileNameRole: b'fileName',
            self.FileSizeRole: b'fileSize',
            self.ModifiedTimeRole: b'modifiedTime',
            self.CaptionsRole: b'captions'
        }

    @Slot()
    def loadNextBatch(self):
        next_batch = self._all_paths_filtered[self._loaded:self._loaded + self._batch_size]
        if not next_batch:
            return
        self.beginInsertRows(QModelIndex(), self._loaded, self._loaded + len(next_batch) - 1)
        for path in next_batch:
            info = ImageInfo(path)
            self._images.append(info)
        self._loaded += len(next_batch)
        self.countChanged.emit()        
        self.endInsertRows()
        if self._loaded < len(self._all_paths_filtered):
            self.canLoadMoreChanged.emit()
        
    @Property(bool, notify=canLoadMoreChanged)
    def canLoadMore(self):
        return self._loaded < len(self._all_paths_filtered)

    # Add this method
    @Slot(str)
    def setSearchText(self, text):
        if self._search_text == text:
            return
        
        self._search_text = text                
        
        self._applyFilter()
        
    @Slot(str)
    def setSearchCaption(self, caption):
        if self._search_caption == caption:
            return
        self._search_caption = caption
        self._applyFilter()
    
    
    @Slot(float)
    def setSearchCaptionSimilarity(self, similarity):
        if self._search_caption_similarity == similarity:
            return
        self._search_caption_similarity = similarity
        self._applyFilter()
        
        
    @Slot(str)
    def setSearchDate(self, start, end):
        if self._search_start_date == start and self._search_end_date == end:
            return
        self._search_start_date = start
        self._search_end_date = end
        self._applyFilter()

    def _applyFilter(self):
        self.beginResetModel()
        if not self._search_text and not self._search_caption and not self._search_start_date and not self._search_end_date:
            self._all_paths_filtered = self._all_paths.copy()
        else:
            paths_filtered = []
            
            for p in self._all_paths:
                if self._search_text and not self._search_text in os.path.basename(p).lower():
                    continue
                
                datetime = QDateTime.fromSecsSinceEpoch(int(os.path.getmtime(p))) 
                if self._search_start_date and datetime < self._search_start_date:
                    continue
                    
                if self._search_end_date and datetime > self._search_end_date:
                    continue
                
                if self._database and self._database.contains(filename=p) and self._search_caption:
                    captions = self._database.find(filename=p)[0]["captions"]
                    is_include = False
                    for c in captions:
                        if self._search_caption in c:
                            is_include = True
                            break
                        else:
                            similarity = get_jaro_similarity(c, self._search_caption)
                            if similarity >= self._search_caption_similarity:
                                is_include = True
                                break
                    
                    if not is_include:
                        continue
                        
                paths_filtered.append(p)
            
            self._all_paths_filtered = paths_filtered
        
        self._images.clear()
        self._loaded = 0
        self.countChanged.emit()
        self.endResetModel()
        self.loadNextBatch()
    
    def setDatabase(self, database):
        self._database = database
        
    count = Property(int, fget=rowCount, notify=countChanged)
        

class FolderItem:
    def __init__(self, folder_path, image_paths):
        self.folder_path = folder_path
        self.image_model = ImageListModel(image_paths)
        self.collapsed = False


class FolderImageModel(QAbstractListModel):
    DirectoryRole = Qt.UserRole + 1
    ImagesRole = Qt.UserRole + 2
    CollapseRole = Qt.UserRole + 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folders = []
        self._database = None

    def rowCount(self, parent=QModelIndex()):
        return len(self._folders)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        folder = self._folders[index.row()]
        if role == self.DirectoryRole:
            return folder.folder_path
        if role == self.ImagesRole:
            return folder.image_model
        if role == self.CollapseRole:
            return folder.collapsed
        return None

    def roleNames(self):
        return {
            self.DirectoryRole: b'directory',
            self.ImagesRole: b'images',
            self.CollapseRole: b'collapsed'
        }

    @Slot(str)
    def toggleCollapse(self, folder_path):
        folder_path = os.path.normpath(folder_path)
        idx = 0
        for folder in self._folders:
            if folder.folder_path == folder_path:
                folder.collapsed = not folder.collapsed
                self.dataChanged.emit(self.index(idx), self.index(idx), [self.CollapseRole])
                break
            idx += 1
        
    @Slot(str)
    def loadFromPath(self, root_path):
        self.beginResetModel()
        self._folders.clear()
        
        for dirpath, _, filenames in os.walk(os.path.normpath(root_path)):
            image_paths = [
                os.path.join(dirpath, f)
                for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if image_paths:
                folderItem = FolderItem(dirpath, image_paths)
                folderItem.image_model.setDatabase(self._database)
                self._folders.append(FolderItem(dirpath, image_paths))

        self.endResetModel()
        
    @Slot(str)
    def setSearchText(self, text):
        for folder in self._folders:
            folder.image_model.setSearchText(text)

    @Slot(str)
    def setSearchCaption(self, caption):
        for folder in self._folders:
            folder.image_model.setSearchCaption(caption)
            
    @Slot(QDateTime, QDateTime)
    def setSearchDate(self, start, end):
        for folder in self._folders:
            folder.image_model.setSearchDate(start, end)
    
    @Slot(float)
    def setSearchCaptionThreshold(self, threshold):
        for folder in self._folders:
            folder.image_model.setSearchCaptionSimilarity(threshold)
            
            
    def setDatabase(self, database):
        for folder in self._folders:
            folder.image_model.setDatabase(database)