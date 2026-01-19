import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const VFX_BRIDGE_NODES = [
    "EXRHotFolderLoader",
    "MatteChannelSplitter", 
    "MetadataDisplay",
    "EXRSaveNode",
    "PreviewMatte",
    "ChannelSelector",
    "EXRToImage",
    "MaskToImage",
    "ColorTransform",
    "DisplayTransform",
    "AOVContactSheet",
    "Text",
];

const ANTHRACITE = "#2d2d2d";
const ANTHRACITE_DARK = "#232323";
const WATERMARK = "rgba(255,255,255,0.06)";

app.registerExtension({
    name: "vfx.bridge.style",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!VFX_BRIDGE_NODES.includes(nodeData.name)) return;
        
        if (nodeData.name === "Text") {
            // ShowText node with proper widget
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                
                // Create a STRING widget for display
                this.showValueWidget = ComfyWidgets["STRING"](this, "output", ["STRING", { multiline: true }], app).widget;
                this.showValueWidget.inputEl.readOnly = true;
                this.showValueWidget.inputEl.style.opacity = 0.8;
                
                // Don't serialize the widget value
                this.showValueWidget.serializeValue = async () => "";
                
                this.color = ANTHRACITE;
                this.bgcolor = ANTHRACITE_DARK;
            };
            
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted ? onExecuted.apply(this, [message]) : undefined;
                if (message && message.text) {
                    this.showValueWidget.value = message.text[0];
                }
            };
            
            const origDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (origDraw) origDraw.apply(this, arguments);
                ctx.save();
                ctx.font = "7px monospace";
                ctx.fillStyle = WATERMARK;
                ctx.textAlign = "right";
                ctx.fillText("peterschings", this.size[0] - 5, this.size[1] - 3);
                ctx.restore();
            };
        } else {
            // All other VFX Bridge nodes - just styling
            const origDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (origDraw) origDraw.apply(this, arguments);
                ctx.save();
                ctx.font = "7px monospace";
                ctx.fillStyle = WATERMARK;
                ctx.textAlign = "right";
                ctx.fillText("peterschings", this.size[0] - 5, this.size[1] - 3);
                ctx.restore();
            };
            
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (origCreated) origCreated.apply(this, arguments);
                this.color = ANTHRACITE;
                this.bgcolor = ANTHRACITE_DARK;
            };
        }
    },
    
    async setup() {
        console.log("[VFX Bridge] Styling loaded");
    }
});
