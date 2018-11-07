# coding: utf-8
import re
from sqlalchemy.dialects.sqlite import DATETIME
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, String, Float, DateTime, text
from sqlalchemy import ForeignKey, BLOB, CLOB, TEXT, NUMERIC
from sqlalchemy.orm import relationship
from renom_img.server.utility.DAO import Base


class Algorithm(Base):

    __tablename__ = 'algorithm'

    algorithm_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)

    relationship("Model")

    def __repr__(self):
        return "<Algorithm(algorithm_id='%s', name='%s')>" % (
            self.algorithm_id,
            self.name
        )


class Task(Base):

    __tablename__ = 'task'

    task_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    deployed_model_id = Column(Integer)

    relationship("Model")

    def __repr__(self):
        return "<Task(task_id='%s', name='%s', deployed_model_id='%s')>" % (
            self.task_id,
            self.name,
            self.deployed_model_id
        )


class Test_Dataset(Base):
    __tablename__ = 'test_dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    description = Column(TEXT)
    test_imgs = Column(CLOB)
    created = Column(DateTime)

    relationship("Dataset")

    def __repr__(self):
        return "<Test_Dataset(test_dataset_id='%s', name='%s')>" % (
            self.test_dataset_id,
            self.name
        )


class Dataset(Base):
    __tablename__ = 'dataset'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    ratio = Column(Float, nullable=False)
    description = Column(TEXT)
    train_data = Column(BLOB)
    valid_data = Column(BLOB)
    class_map = Column(BLOB)
    class_tag_list = Column(BLOB)
    created = Column(DateTime(timezone=True), server_default=func.now())
    test_dataset_id = Column(
        Integer,
        ForeignKey('test_dataset.id', ondelete='CASCADE')
    )

    def __repr__(self):
        line = """
                <Dataset(
                    id='%s',
                    name='%s',
                    ratio='%s',
                    description='%s',
                    train_imgs='%s',
                    valid_imgs='%s'
                    class_map='%s',
                    class_tag_list='%s'
                    created='%s'
                    test_dataset_id='%s',
                )>
                """
        return line % (
            self.id,
            self.name,
            self.ratio,
            self.description,
            self.train_data,
            self.valid_data,
            self.class_map,
            self.class_tag_list,
            self.created,
            self.test_dataset_id
        )


class Model(Base):

    __tablename__ = 'model'

    # Must be given
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey('task.task_id'))
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'))
    algorithm_id = Column(Integer,  ForeignKey('algorithm.algorithm_id'))
    hyper_parameters = Column(BLOB)

    # Modified during training.
    state = Column(Integer, server_default=text('0'))
    train_loss_list = Column(BLOB, nullable=True)
    validation_loss_list = Column(BLOB, nullable=True)
    best_epoch = Column(Integer, server_default=text('0'))
    best_epoch_iou = Column(NUMERIC, server_default=text('0'))
    best_epoch_map = Column(NUMERIC, server_default=text('0'))
    best_epoch_validation_reuslt = Column(BLOB, nullable=True)
    best_epoch_weight = Column(TEXT)
    last_epoch = Column(Integer, server_default=text('0'))
    last_weight = Column(TEXT)
    last_batch = Column(Integer, server_default=text('0'))
    last_train_loss = Column(NUMERIC, server_default=text('0'))
    total_batch = Column(Integer, server_default=text('0'))
    running_state = Column(Integer, server_default=text('0'))

    # Dates
    created = Column(DateTime(timezone=True), server_default=func.now())
    updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __init__(self, *arg, **kwargs):
        # Register path.
        last_weight_name = "last_model_{}.h5".format(11)
        self.last_weight = last_weight_name
        best_weight_name = "best_model_{}.h5".format(11)
        self.best_epoch_weight = best_weight_name

    def __repr__(self):
        return "<Model(id='%s', task_id='%s')>" % (
            self.id,
            self.task_id
        )
