"""
log utils
"""
import os
import time
import sys

USE_TENSORBOARD = False


class Logger:
    """Create a summary writer logging to log_dir."""

    def __init__(self, opt):
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        args = {}
        for name in dir(opt):
            if not name.startswith('_'):
                args[name] = getattr(opt, name)
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = opt.save_dir + '/logs_{}'.format(time_str)
        if not os.path.exists(os.path.dirname(log_dir)):
            os.mkdir(os.path.dirname(log_dir))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        try:
            os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
        except IOError:
            pass
        else:
            pass
        self.start_line = True

    def write(self, txt):
        """write"""
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        """close"""
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)
