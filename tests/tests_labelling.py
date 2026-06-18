from ml_on_apx.labelling import Labels, Label
import unittest


class TestLabels(unittest.TestCase):
    def test_labelling__test_label_eq(self):
        label1 = Label("a")
        label2 = Label("a")
        self.assertEqual(label1, label2)

    def test_labelling__instantiation(self):
        Labels(set([Label("a"), Label("b")]))

    def test_labelling__contains(self):
        labels = Labels(set([Label("a"), Label("b")]))
        self.assertTrue(Label("a") in labels)
        self.assertTrue(Label("b") in labels)
        self.assertFalse(Label("c") in labels)

    def test_labelling__get_item(self):
        labels = Labels(set([Label("a"), Label("b")]))
        self.assertEqual(0, labels[Label("a")])
        self.assertEqual(1, labels[Label("b")])
        with self.assertRaises(KeyError):
            labels[Label("c")]

    def test_labelling__test_labels_eq(self):
        labels1 = Labels(set([Label("a"), Label("b")]))
        labels2 = Labels(set([Label("b"), Label("a")]))
        self.assertEqual(labels1, labels2)
