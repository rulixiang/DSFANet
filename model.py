
import tensorflow as tf

#from tensorflow.python.ops.gen_array_ops import transpose
#from tensorflow.python.platform.tf_logging import log


class DSFANet(object):
    def __init__(self, num=None):
        self.num = num
        self.output_num = 6
        self.hidden_num = 128
        self.layers = 2
        self.reg = 1e-4
        self.activation = tf.nn.softsign
        self.init = tf.initializers.he_normal()

    def DSFA(self, X, Y):

        #m, n = tf.shape(X)
        X_hat = X - tf.reduce_mean(X, axis=0)
        Y_hat = Y - tf.reduce_mean(Y, axis=0)

        differ = X_hat - Y_hat

        A = tf.matmul(differ, differ, transpose_a=True)
        A = A/self.num

        Sigma_XX = tf.matmul(X_hat, X_hat, transpose_a=True)
        Sigma_XX = Sigma_XX / self.num + self.reg * tf.eye(self.output_num)
        Sigma_YY = tf.matmul(Y_hat, Y_hat, transpose_a=True)
        Sigma_YY = Sigma_YY / self.num + self.reg * tf.eye(self.output_num)

        B = (Sigma_XX+Sigma_YY)/2

        # For numerical stability.
        D_B, V_B = tf.self_adjoint_eig(B)
        idx = tf.where(D_B > 1e-12)[:, 0]
        D_B = tf.gather(D_B, idx)
        V_B = tf.gather(V_B, idx, axis=1)
        B_inv = tf.matmul(tf.matmul(V_B, tf.diag(tf.reciprocal(D_B))), tf.transpose(V_B))
        ##

        Sigma = tf.matmul(B_inv, A)
        loss = tf.trace(tf.matmul(Sigma, Sigma))

        return loss

    def forward(self, X, Y):

        for k in range(self.layers):
            X = tf.layers.dense(inputs=X, units=self.hidden_num, activation=self.activation, use_bias=True,)
            Y = tf.layers.dense(inputs=Y, units=self.hidden_num, activation=self.activation, use_bias=True,)

        self.X_ = tf.layers.dense(inputs=X, units=self.output_num, activation=self.activation, use_bias=True,)
        self.Y_ = tf.layers.dense(inputs=Y, units=self.output_num, activation=self.activation, use_bias=True,)

        loss = self.DSFA(self.X_, self.Y_)

        return loss
