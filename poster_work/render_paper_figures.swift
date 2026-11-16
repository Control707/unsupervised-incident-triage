import Foundation
import AppKit
import PDFKit

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let output = root.appendingPathComponent("poster_work/paper_figures")
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
for name in ["v4_figure2_r1_split", "v3_figure3_r1_r3_rolling", "v4_figure4_pooling_cancellation", "v3_figure5_payoff_triage"] {
    let source = root.appendingPathComponent("figures/\(name).pdf")
    guard let document = PDFDocument(url: source), let page = document.page(at: 0) else { fatalError("Cannot read \(source)") }
    let bounds = page.bounds(for: .mediaBox)
    let scale: CGFloat = 6
    let width = Int(ceil(bounds.width * scale)), height = Int(ceil(bounds.height * scale))
    guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0), let ctx = NSGraphicsContext(bitmapImageRep: bitmap) else { fatalError("No bitmap") }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = ctx
    let cg = ctx.cgContext
    cg.setFillColor(NSColor.white.cgColor)
    cg.fill(CGRect(x: 0, y: 0, width: width, height: height))
    cg.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: cg)
    NSGraphicsContext.restoreGraphicsState()
    guard let data = bitmap.representation(using: .png, properties: [:]) else { fatalError("No PNG") }
    let dest = output.appendingPathComponent("\(name).png")
    try data.write(to: dest)
    print("\(dest.path): \(width) x \(height)")
}
