#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase, main
from dag_builder import *
from ir.values import *
from ir.types import *


class ParseExpressionTest(TestCase):
    pass


class ConstantIntTest(TestCase):

    def test_eq(self):
        value1 = ConstantInt(100, i32)
        value2 = ConstantInt(100, i32)

        self.assertEqual(value1, value2)

        value1 = ConstantInt(10, i32)
        value2 = ConstantInt(100, i32)

        self.assertNotEqual(value1, value2)

        value1 = ConstantInt(100, i32)
        value2 = ConstantInt(100, i16)

        self.assertNotEqual(value1, value2)

    def test_hash(self):
        d = set([
            ConstantInt(100, i32)
        ])
        value = ConstantInt(100, i32)

        self.assertTrue(value in d)

        value = ConstantInt(100, i64)

        self.assertFalse(value in d)


if __name__ == '__main__':
    main()
