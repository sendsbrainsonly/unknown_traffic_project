# VS Code 图片显示修复

- 日期：2026-10-16；所属聊天：`01a144b7-de8c-7521-a7ae-d8575754129d`。
- 状态：`complete`。
- 问题复现：在用户实际打开的 VS Code `1.md — unknown traffic` Markdown 预览中，图4-6和图4-7显示破图占位符；七张原始PNG本身均可解码。
- 修复：将正文七处本机绝对图片路径改为相对 `paper/md/ch4/1.md` 的 `../../picture/ch4/20261016-experiment-figures-v1/render-v4/…png`。移除对机器根路径、路径中空格及尖括号写法的依赖；正文其余内容及原图均不变。
- 实际验证：修改后同一VS Code预览自动更新，截图中原图4-6占位符已变为完整三面板数据图。无需关闭预览安全限制或修改编辑器设置。未单独测试Cursor界面。
- 全量路径／数据核验：`verification.json`为此次新回执；7个图片引用均解析至正确PNG，173个指标、14个标准差、128处正文数值及21份正式图件哈希全部通过。
- 修复前文件：`manuscript-before.md`、`verify_figures-before.py`；新核验器同时支持相对路径和原绝对路径，且仍要求全部图件指向指定正式渲染目录。
- 额外一致性检查：仅7个图片引用行变化；归一化图片路径后，修复前后整篇文稿完全一致。
- 界面探测：确认编辑器前，两次Typora读取超时；用户随后明确使用VS Code／Cursor，实际VS Code读取与预览验证成功。

复核命令（从项目根执行，使用新的回执文件名，避免覆盖历史）：

```bash
python3 paper/picture/ch4/20261016-experiment-figures-v1/verify_figures.py paper/picture/ch4/20261016-experiment-figures-v1/render-v4 --output paper/picture/ch4/20261016-experiment-figures-v1/display-fix-20261016/verification-repeat-01.json
```

移动文件时需保持 `paper/md/ch4/` 与 `paper/picture/` 的相对目录关系；可整体移动 `paper/` 目录。
