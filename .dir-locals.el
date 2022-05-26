;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((python-mode . ((eval . (setq eglot-workspace-configuration
                               `((:python
                                  :venvPath ,(expand-absolute-name "~/.local/share/conda/envs")
                                  :analysis (:diagnosticMode
                                             "openFilesOnly"
                                             :stubPath
                                             ,(expand-absolute-name "~/.local/lib/python-type-stubs")))))))))
