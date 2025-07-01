import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch

# 导入被测试的类
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.strongsort.sort.detection import Detection


class TestStrongSortUpdate(unittest.TestCase):
    """StrongSort update方法的综合单元测试"""

    def setUp(self):
        """测试前的设置"""
        # Mock所有外部依赖
        with patch('boxmot.trackers.strongsort.strongsort.ReidAutoBackend') as mock_reid, \
             patch('boxmot.trackers.strongsort.strongsort.Tracker') as mock_tracker_class, \
             patch('boxmot.trackers.strongsort.strongsort.get_cmc_method') as mock_cmc_method:
            
            # 配置mock对象
            mock_model = Mock()
            mock_model.get_features.return_value = np.random.rand(0, 128)
            mock_reid.return_value.model = mock_model
            
            mock_tracker = Mock()
            mock_tracker.tracks = []
            mock_tracker_class.return_value = mock_tracker
            
            mock_cmc = Mock()
            mock_cmc.apply.return_value = np.eye(3)
            mock_cmc_method.return_value = lambda: mock_cmc
            
            # 创建StrongSort实例
            self.strongsort = StrongSort(
                reid_weights=Path("dummy.pt"),
                device=torch.device("cpu"),
                half=False,
                per_class=False,
                min_conf=0.5
            )
            
            # 保存mock对象的引用
            self.mock_model = self.strongsort.model
            self.mock_tracker = self.strongsort.tracker
            self.mock_cmc = self.strongsort.cmc

    def test_update_empty_detections(self):
        """测试空检测数组"""
        dets = np.zeros((0, 7))
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock相关方法
        self.mock_model.get_features.return_value = np.empty((0, 128))
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 0)
        self.mock_tracker.predict.assert_called_once()
        self.mock_tracker.update.assert_called_once()

    def test_update_single_detection(self):
        """测试单个检测"""
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, 1]  # x1, y1, x2, y2, conf, cls, tracker_id
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock一个确认的轨迹
        mock_track = Mock()
        mock_track.is_confirmed.return_value = True
        mock_track.time_since_update = 0
        mock_track.to_tlbr.return_value = (100, 100, 200, 200)
        mock_track.id = 1
        mock_track.conf = 0.8
        mock_track.cls = 0
        mock_track.det_ind = 0
        mock_track.camera_update = Mock()
        
        self.mock_tracker.tracks = [mock_track]
        
        # Mock相关方法
        self.mock_cmc.apply.return_value = np.eye(3)
        self.mock_model.get_features.return_value = np.random.rand(1, 128)
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 1)  # 一个轨迹
        self.assertEqual(result.shape[1], 8)  # 8个字段: x1,y1,x2,y2,id,conf,cls,det_ind
        
        # 验证方法调用
        self.mock_tracker.predict.assert_called_once()
        self.mock_tracker.update.assert_called_once()
        mock_track.camera_update.assert_called_once()

    def test_update_multiple_detections(self):
        """测试多个检测"""
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, 1],
            [300, 300, 400, 400, 0.9, 1, 2],
            [500, 500, 600, 600, 0.7, 0, 3]
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock多个确认的轨迹
        mock_tracks = []
        for i in range(3):
            mock_track = Mock()
            mock_track.is_confirmed.return_value = True
            mock_track.time_since_update = 0
            mock_track.to_tlbr.return_value = (100 + i*200, 100 + i*200, 200 + i*200, 200 + i*200)
            mock_track.id = i + 1
            mock_track.conf = 0.8 + i*0.05
            mock_track.cls = i % 2
            mock_track.det_ind = i
            mock_track.camera_update = Mock()
            mock_tracks.append(mock_track)
        
        self.mock_tracker.tracks = mock_tracks
        
        # Mock相关方法
        self.mock_cmc.apply.return_value = np.eye(3)
        self.mock_model.get_features.return_value = np.random.rand(3, 128)
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 3)  # 三个轨迹
        self.assertEqual(result.shape[1], 8)  # 8个字段
        
        # 验证所有轨迹都调用了camera_update
        for track in mock_tracks:
            track.camera_update.assert_called_once()

    def test_update_low_confidence_filtering(self):
        """测试低置信度检测过滤"""
        dets = np.array([
            [100, 100, 200, 200, 0.3, 0, 1],  # 低置信度，应被过滤
            [300, 300, 400, 400, 0.8, 1, 2]   # 高置信度，应保留
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock相关方法
        self.mock_model.get_features.return_value = np.random.rand(1, 128)  # 只有一个高置信度检测
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证tracker.update只接收到一个检测（高置信度的）
        self.mock_tracker.update.assert_called_once()
        detections_arg = self.mock_tracker.update.call_args[0][0]
        self.assertEqual(len(detections_arg), 1)  # 只有一个检测通过置信度过滤

    def test_update_with_tracker_id_mapping(self):
        """测试带有tracker_id映射的情况"""
        target_id = 999
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, target_id]
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock一个确认的轨迹
        mock_track = Mock()
        mock_track.is_confirmed.return_value = True
        mock_track.time_since_update = 0
        mock_track.to_tlbr.return_value = (100, 100, 200, 200)
        mock_track.id = 1  # 原始ID
        mock_track.conf = 0.8
        mock_track.cls = 0
        mock_track.det_ind = 0
        mock_track.camera_update = Mock()
        
        self.mock_tracker.tracks = [mock_track]
        
        # Mock相关方法
        self.mock_cmc.apply.return_value = np.eye(3)
        self.mock_model.get_features.return_value = np.random.rand(1, 128)
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证ID被更新为目标ID
        self.assertEqual(mock_track.id, target_id)
        
        # 验证结果包含更新后的ID
        self.assertIsInstance(result, np.ndarray)
        if len(result) > 0:
            self.assertEqual(result[0, 4], target_id)  # ID字段应该是目标ID

    def test_update_unconfirmed_tracks_filtered(self):
        """测试未确认的轨迹被过滤"""
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, 1]
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock一个未确认的轨迹和一个确认的轨迹
        unconfirmed_track = Mock()
        unconfirmed_track.is_confirmed.return_value = False
        unconfirmed_track.time_since_update = 0
        unconfirmed_track.camera_update = Mock()
        
        confirmed_track = Mock()
        confirmed_track.is_confirmed.return_value = True
        confirmed_track.time_since_update = 0
        confirmed_track.to_tlbr.return_value = (100, 100, 200, 200)
        confirmed_track.id = 1
        confirmed_track.conf = 0.8
        confirmed_track.cls = 0
        confirmed_track.det_ind = 0
        confirmed_track.camera_update = Mock()
        
        self.mock_tracker.tracks = [unconfirmed_track, confirmed_track]
        
        # Mock相关方法
        self.mock_cmc.apply.return_value = np.eye(3)
        self.mock_model.get_features.return_value = np.random.rand(1, 128)
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证结果只包含确认的轨迹
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 1)  # 只有一个确认的轨迹

    def test_update_old_tracks_filtered(self):
        """测试过旧的轨迹被过滤"""
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, 1]
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock一个过旧的轨迹和一个新的轨迹
        old_track = Mock()
        old_track.is_confirmed.return_value = True
        old_track.time_since_update = 2  # 大于等于1，应被过滤
        old_track.camera_update = Mock()
        
        new_track = Mock()
        new_track.is_confirmed.return_value = True
        new_track.time_since_update = 0  # 小于1，应保留
        new_track.to_tlbr.return_value = (100, 100, 200, 200)
        new_track.id = 1
        new_track.conf = 0.8
        new_track.cls = 0
        new_track.det_ind = 0
        new_track.camera_update = Mock()
        
        self.mock_tracker.tracks = [old_track, new_track]
        
        # Mock相关方法
        self.mock_cmc.apply.return_value = np.eye(3)
        self.mock_model.get_features.return_value = np.random.rand(1, 128)
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证结果只包含新的轨迹
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 1)  # 只有一个新的轨迹

    def test_update_with_custom_embeddings(self):
        """测试使用自定义嵌入向量"""
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, 1]
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        custom_embs = np.random.rand(1, 128)
        
        # Mock相关方法
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img, embs=custom_embs)
        
        # 验证model.get_features没有被调用（因为提供了自定义嵌入）
        self.mock_model.get_features.assert_not_called()
        
        # 验证tracker.update被调用
        self.mock_tracker.update.assert_called_once()

    def test_update_invalid_input_types(self):
        """测试无效输入类型"""
        # 测试非numpy数组的dets
        with self.assertRaises(AssertionError):
            self.strongsort.update([[100, 100, 200, 200, 0.8, 0, 1]], np.zeros((480, 640, 3)))
        
        # 测试非numpy数组的img
        with self.assertRaises(AssertionError):
            self.strongsort.update(np.array([[100, 100, 200, 200, 0.8, 0, 1]]), [[[0]]])

    def test_update_invalid_dets_dimensions(self):
        """测试无效的检测维度"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 测试一维数组
        with self.assertRaises(AssertionError):
            self.strongsort.update(np.array([100, 100, 200, 200, 0.8, 0, 1]), img)
        
        # 测试错误的第二维长度
        with self.assertRaises(AssertionError):
            self.strongsort.update(np.array([[100, 100, 200, 200, 0.8, 0]]), img)  # 缺少tracker_id

    def test_update_no_existing_tracks(self):
        """测试没有现有轨迹的情况"""
        dets = np.array([
            [100, 100, 200, 200, 0.8, 0, 1]
        ])
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 确保没有现有轨迹
        self.mock_tracker.tracks = []
        
        # Mock相关方法
        self.mock_model.get_features.return_value = np.random.rand(1, 128)
        self.mock_tracker.predict = Mock()
        self.mock_tracker.update = Mock()
        
        result = self.strongsort.update(dets, img)
        
        # 验证CMC不被调用（因为没有现有轨迹）
        self.mock_cmc.apply.assert_not_called()
        
        # 验证结果为空（因为没有确认的轨迹）
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()