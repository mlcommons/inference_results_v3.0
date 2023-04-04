#include "dnnl.hpp"
#include "dnnl_types.h"
#include <cstring>
#include <iostream>
#include "oneDNN_var_init.cpp"


#include "backbone_8.cpp"


// Post Backbone Initialization

static dnnl::memory::dims fc_src_tz_ = {8, 2048, 1, 1};
static dnnl::memory::dims fc_weights_tz_ = {1000, 2048, 1, 1};
static dnnl::memory::dims fc_bias_tz_ = {1000};
static dnnl::memory::dims fc_dst_tz_ = {8, 1000};

static dnnl::memory::dims avgpool_src_tz_ = {8, 2048, 7, 7};
static dnnl::memory::dims avgpool_dst_tz_ = {8, 2048, 1, 1};
static dnnl::memory::dims avgpool_kernel_sz_ = {7,7};
static dnnl::memory::dims avgpool_strides_sz_ = {2,2};
static dnnl::memory::dims avgpool_padding_sz_ = {0,0};

// dnnl::memory conv_weights_memory, conv_bias_memory;

// dnnl::memory fc_weights_memory;
// dnnl::memory fc_bias_memory;



static void prepareOneDNN(float* conv1_weight, float* conv1_bias,float* fc_weight, float* fc_bias){
    if (init_onednn){
      return;
    }

    auto user_weights_memory = dnnl::memory({{conv_weights_tz_stg1_}, dt::f32, tag::oihw}, eng_dnn_);

    memcpy(user_weights_memory.get_data_handle(), conv1_weight, Start_Out_C*Start_In_C*Start_W_H*Start_W_W*sizeof(float));
    auto user_bias_memory = dnnl::memory({{conv_bias_tz_stg1_}, dt::f32, tag::x}, eng_dnn_);
    memcpy(user_bias_memory.get_data_handle(), conv1_bias, Start_Out_C*sizeof(float));

    conv_weights_memory = dnnl::memory({{conv_weights_tz_stg1_}, dt::s8, tag::Adcb16a}, eng_dnn_);
    conv_bias_memory = dnnl::memory({{conv_bias_tz_stg1_}, dt::f32, tag::a}, eng_dnn_);

    const float post_scale = 1.f;
    const std::vector<float> weight_scales = {post_scale/0.0006816235836595297f,
                                            post_scale/0.0005305774975568056f,
                                            post_scale/0.00016325304750353098f,
                                            post_scale/0.0002625430643092841f,
                                            post_scale/0.001501173130236566f,
                                            post_scale/0.0002946545137092471f,
                                            post_scale/0.0013540860963985324f,
                                            post_scale/0.0012125875800848007f,
                                            post_scale/0.0018035992980003357f,
                                            post_scale/0.00025723729049786925f,
                                            post_scale/0.001094487844966352f,
                                            post_scale/0.00026009505381807685f,
                                            post_scale/0.0008863380062393844f,
                                            post_scale/1.1920928955078125e-07f,
                                            post_scale/0.0024081964511424303f,
                                            post_scale/0.0015865263994783163f,
                                            post_scale/0.0004841082845814526f,
                                            post_scale/0.001560483011417091f,
                                            post_scale/0.0025746244937181473f,
                                            post_scale/0.000310599833028391f,
                                            post_scale/0.002417005831375718f,
                                            post_scale/0.0015911447117105126f,
                                            post_scale/0.0022614472545683384f,
                                            post_scale/0.00043826879118569195f,
                                            post_scale/0.0007691492792218924f,
                                            post_scale/0.00020982741261832416f,
                                            post_scale/0.0012938278960064054f,
                                            post_scale/0.0012921467423439026f,
                                            post_scale/0.0004248563782311976f,
                                            post_scale/0.00024047742772381753f,
                                            post_scale/0.0005666522774845362f,
                                            post_scale/0.00013184541603550315f,
                                            post_scale/0.0002759526832960546f,
                                            post_scale/0.0038828933611512184f,
                                            post_scale/0.00014129547344055027f,
                                            post_scale/0.00030801387038081884f,
                                            post_scale/0.0002745217061601579f,
                                            post_scale/0.00030328892171382904f,
                                            post_scale/0.0010830751853063703f,
                                            post_scale/0.0011779471533372998f,
                                            post_scale/0.00022047704260330647f,
                                            post_scale/0.001532222144305706f,
                                            post_scale/0.0002601104788482189f,
                                            post_scale/0.00037946386146359146f,
                                            post_scale/0.0001750480878399685f,
                                            post_scale/0.0008486405131407082f,
                                            post_scale/0.0015950539382174611f,
                                            post_scale/0.0004155198694206774f,
                                            post_scale/0.0008059836691245437f,
                                            post_scale/0.0017546059098094702f,
                                            post_scale/0.001436483347788453f,
                                            post_scale/0.000660675170365721f,
                                            post_scale/0.00042748029227368534f,
                                            post_scale/0.0012573313433676958f,
                                            post_scale/0.0011386079713702202f,
                                            post_scale/0.000577236816752702f,
                                            post_scale/0.0005883735720999539f,
                                            post_scale/0.00015980478201527148f,
                                            post_scale/0.00045653333654627204f,
                                            post_scale/0.00031122786458581686f,
                                            post_scale/0.0006546518416143954f,
                                            post_scale/0.0009724263800308108f,
                                            post_scale/0.00027550148661248386f,
                                            post_scale/0.0025447772350162268f};

    const float in_scale = 0.02070588245987892;
    const std::vector<float> conv_scales = {in_scale*0.0006816235836595297f,
                                            in_scale*0.0005305774975568056f,
                                            in_scale*0.00016325304750353098f,
                                            in_scale*0.0002625430643092841f,
                                            in_scale*0.001501173130236566f,
                                            in_scale*0.0002946545137092471f,
                                            in_scale*0.0013540860963985324f,
                                            in_scale*0.0012125875800848007f,
                                            in_scale*0.0018035992980003357f,
                                            in_scale*0.00025723729049786925f,
                                            in_scale*0.001094487844966352f,
                                            in_scale*0.00026009505381807685f,
                                            in_scale*0.0008863380062393844f,
                                            in_scale*1.1920928955078125e-07f,
                                            in_scale*0.0024081964511424303f,
                                            in_scale*0.0015865263994783163f,
                                            in_scale*0.0004841082845814526f,
                                            in_scale*0.001560483011417091f,
                                            in_scale*0.0025746244937181473f,
                                            in_scale*0.000310599833028391f,
                                            in_scale*0.002417005831375718f,
                                            in_scale*0.0015911447117105126f,
                                            in_scale*0.0022614472545683384f,
                                            in_scale*0.00043826879118569195f,
                                            in_scale*0.0007691492792218924f,
                                            in_scale*0.00020982741261832416f,
                                            in_scale*0.0012938278960064054f,
                                            in_scale*0.0012921467423439026f,
                                            in_scale*0.0004248563782311976f,
                                            in_scale*0.00024047742772381753f,
                                            in_scale*0.0005666522774845362f,
                                            in_scale*0.00013184541603550315f,
                                            in_scale*0.0002759526832960546f,
                                            in_scale*0.0038828933611512184f,
                                            in_scale*0.00014129547344055027f,
                                            in_scale*0.00030801387038081884f,
                                            in_scale*0.0002745217061601579f,
                                            in_scale*0.00030328892171382904f,
                                            in_scale*0.0010830751853063703f,
                                            in_scale*0.0011779471533372998f,
                                            in_scale*0.00022047704260330647f,
                                            in_scale*0.001532222144305706f,
                                            in_scale*0.0002601104788482189f,
                                            in_scale*0.00037946386146359146f,
                                            in_scale*0.0001750480878399685f,
                                            in_scale*0.0008486405131407082f,
                                            in_scale*0.0015950539382174611f,
                                            in_scale*0.0004155198694206774f,
                                            in_scale*0.0008059836691245437f,
                                            in_scale*0.0017546059098094702f,
                                            in_scale*0.001436483347788453f,
                                            in_scale*0.000660675170365721f,
                                            in_scale*0.00042748029227368534f,
                                            in_scale*0.0012573313433676958f,
                                            in_scale*0.0011386079713702202f,
                                            in_scale*0.000577236816752702f,
                                            in_scale*0.0005883735720999539f,
                                            in_scale*0.00015980478201527148f,
                                            in_scale*0.00045653333654627204f,
                                            in_scale*0.00031122786458581686f,
                                            in_scale*0.0006546518416143954f,
                                            in_scale*0.0009724263800308108f,
                                            in_scale*0.00027550148661248386f,
                                            in_scale*0.0025447772350162268f};

    const std::vector<float> bias_scales = {1/(in_scale*0.0006816235836595297f),
                                            1/(in_scale*0.0005305774975568056f),
                                            1/(in_scale*0.00016325304750353098f),
                                            1/(in_scale*0.0002625430643092841f),
                                            1/(in_scale*0.001501173130236566f),
                                            1/(in_scale*0.0002946545137092471f),
                                            1/(in_scale*0.0013540860963985324f),
                                            1/(in_scale*0.0012125875800848007f),
                                            1/(in_scale*0.0018035992980003357f),
                                            1/(in_scale*0.00025723729049786925f),
                                            1/(in_scale*0.001094487844966352f),
                                            1/(in_scale*0.00026009505381807685f),
                                            1/(in_scale*0.0008863380062393844f),
                                            1/(in_scale*1.1920928955078125e-07f),
                                            1/(in_scale*0.0024081964511424303f),
                                            1/(in_scale*0.0015865263994783163f),
                                            1/(in_scale*0.0004841082845814526f),
                                            1/(in_scale*0.001560483011417091f),
                                            1/(in_scale*0.0025746244937181473f),
                                            1/(in_scale*0.000310599833028391f),
                                            1/(in_scale*0.002417005831375718f),
                                            1/(in_scale*0.0015911447117105126f),
                                            1/(in_scale*0.0022614472545683384f),
                                            1/(in_scale*0.00043826879118569195f),
                                            1/(in_scale*0.0007691492792218924f),
                                            1/(in_scale*0.00020982741261832416f),
                                            1/(in_scale*0.0012938278960064054f),
                                            1/(in_scale*0.0012921467423439026f),
                                            1/(in_scale*0.0004248563782311976f),
                                            1/(in_scale*0.00024047742772381753f),
                                            1/(in_scale*0.0005666522774845362f),
                                            1/(in_scale*0.00013184541603550315f),
                                            1/(in_scale*0.0002759526832960546f),
                                            1/(in_scale*0.0038828933611512184f),
                                            1/(in_scale*0.00014129547344055027f),
                                            1/(in_scale*0.00030801387038081884f),
                                            1/(in_scale*0.0002745217061601579f),
                                            1/(in_scale*0.00030328892171382904f),
                                            1/(in_scale*0.0010830751853063703f),
                                            1/(in_scale*0.0011779471533372998f),
                                            1/(in_scale*0.00022047704260330647f),
                                            1/(in_scale*0.001532222144305706f),
                                            1/(in_scale*0.0002601104788482189f),
                                            1/(in_scale*0.00037946386146359146f),
                                            1/(in_scale*0.0001750480878399685f),
                                            1/(in_scale*0.0008486405131407082f),
                                            1/(in_scale*0.0015950539382174611f),
                                            1/(in_scale*0.0004155198694206774f),
                                            1/(in_scale*0.0008059836691245437f),
                                            1/(in_scale*0.0017546059098094702f),
                                            1/(in_scale*0.001436483347788453f),
                                            1/(in_scale*0.000660675170365721f),
                                            1/(in_scale*0.00042748029227368534f),
                                            1/(in_scale*0.0012573313433676958f),
                                            1/(in_scale*0.0011386079713702202f),
                                            1/(in_scale*0.000577236816752702f),
                                            1/(in_scale*0.0005883735720999539f),
                                            1/(in_scale*0.00015980478201527148f),
                                            1/(in_scale*0.00045653333654627204f),
                                            1/(in_scale*0.00031122786458581686f),
                                            1/(in_scale*0.0006546518416143954f),
                                            1/(in_scale*0.0009724263800308108f),
                                            1/(in_scale*0.00027550148661248386f),
                                            1/(in_scale*0.0025447772350162268f)};

    

    const int weight_mask = 1;
    const int bias_mask = 1;
    const int conv_mask = 2;

    dnnl::primitive_attr weight_attr;
    weight_attr.set_output_scales(weight_mask, weight_scales);
    auto weight_reorder_pd = dnnl::reorder::primitive_desc(eng_dnn_, user_weights_memory.get_desc(),
                                                           eng_dnn_, conv_weights_memory.get_desc(), weight_attr);
    auto weight_reorder = dnnl::reorder(weight_reorder_pd);
    weight_reorder.execute(dnn_strm_, user_weights_memory, conv_weights_memory);

    dnnl::primitive_attr bias_attr;
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd = dnnl::reorder::primitive_desc(eng_dnn_, user_bias_memory.get_desc(),
                                                         eng_dnn_, conv_bias_memory.get_desc(), bias_attr);
    auto bias_reorder = dnnl::reorder(bias_reorder_pd);
    bias_reorder.execute(dnn_strm_, user_bias_memory, conv_bias_memory);

    auto conv_src_md = dnnl::memory::desc({conv_src_tz_stg1_}, dt::s8, tag::any);
    auto conv_bias_md = dnnl::memory::desc({conv_bias_tz_stg1_}, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc({conv_weights_tz_stg1_}, dt::s8, tag::Adcb16a);
    auto conv_dst_md = dnnl::memory::desc({conv_dst_tz_stg1_}, dt::s8, tag::nhwc);

    memcpy(conv_weights_ptr_stg1_, conv_weights_memory.get_data_handle(), conv_weights_md.get_size());
    memcpy(conv_bias_ptr_stg1_, conv_bias_memory.get_data_handle(), 64*sizeof(float));

    auto conv_desc = dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, conv_src_md, conv_weights_md, 
            conv_bias_md, conv_dst_md, conv_strides_stg1_, conv_padding_stg1_, conv_padding_stg1_);
    
    dnnl::primitive_attr conv_attr;
    conv_attr.set_output_scales(conv_mask, conv_scales);
    conv_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const float ops_scale = 1./0.05720944702625275;
    const float ops_alpha = 0.f; // SKip?
    const float ops_beta = 0.f;
    dnnl::post_ops ops;
    ops.append_eltwise(ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, conv_attr, eng_dnn_);
    scratchpad_md_prim_ = conv_prim_desc.scratchpad_desc();

    conv_forward_prim_ = dnnl::convolution_forward(conv_prim_desc);

    auto pool_dst_md = dnnl::memory::desc({maxpool_dst_tz_}, dt::s8, tag::any);
    auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_max, conv_dst_md, pool_dst_md,
            maxpool_strides_sz_, maxpool_kernel_sz_, maxpool_padding_sz_, maxpool_padding_sz_);
    auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, eng_dnn_);
    
    pool_forward_prim_ = dnnl::pooling_forward(pool_pd);

    // Post Backbone

    dnnl::memory avg_pool_dst_memory_ = dnnl::memory({{avgpool_dst_tz_}, dt::s8, tag::nhwc}, eng_dnn_);
    dnnl::memory fc_src_memory_ = dnnl::memory({{fc_src_tz_}, dt::s8, tag::nhwc}, eng_dnn_);
      
    const int fc_weight_mask = 1;
    const int fc_bias_mask = 1;
    const int fc_mask = 2;
    
    auto avg_pool_src_md = dnnl::memory::desc({avgpool_src_tz_}, dt::s8, tag::nhwc); 
    auto avg_pool_dst_md = dnnl::memory::desc({avgpool_dst_tz_}, dt::s8, tag::nhwc);
   
    auto avg_pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_avg_exclude_padding, avg_pool_src_md, avg_pool_dst_md,
            avgpool_strides_sz_, avgpool_kernel_sz_, avgpool_padding_sz_, avgpool_padding_sz_);
    dnnl::primitive_attr avg_pool_attr;
    avg_pool_attr.set_output_scales(0,{0.18475806713104248});
    auto avg_pool_pd = dnnl::pooling_forward::primitive_desc(avg_pool_desc, eng_dnn_);


    avg_pool_forward_prim_ = dnnl::pooling_forward(avg_pool_pd);

    auto fc_src_md = dnnl::memory::desc({fc_src_tz_}, dt::s8, tag::any);
   
    auto fc_weights_md = dnnl::memory::desc({fc_weights_tz_}, dt::s8, tag::any);
 
    auto fc_bias_md = dnnl::memory::desc({fc_bias_tz_}, dt::f32, tag::any);
   
    auto fc_dst_md = dnnl::memory::desc({fc_dst_tz_}, dt::f32, tag::any);

    

    dnnl::primitive_attr fc_attr;
    fc_attr.set_output_scales(0,{1/0.18475806713104248});
   
    auto input_pd = dnnl::reorder::primitive_desc(eng_dnn_, avg_pool_dst_memory_.get_desc(),
                                                           eng_dnn_, fc_src_memory_.get_desc(),fc_attr);
    auto input_reorder = dnnl::reorder(input_pd);
    
    reorder_scratchpad_md_8 = input_pd.scratchpad_desc();
  
    fc_input_reorder = input_reorder;

    
    auto fc_user_weights_memory = dnnl::memory({{fc_weights_tz_}, dt::f32, tag::oihw}, eng_dnn_);
    memcpy(fc_user_weights_memory.get_data_handle(), fc_weight, 2048*1000*sizeof(float));
    auto fc_user_bias_memory = dnnl::memory({fc_bias_tz_, dt::f32, tag::x}, eng_dnn_);
    memcpy(fc_user_bias_memory.get_data_handle(), fc_bias, 1000*sizeof(float));
    
    const float post_scale_fc = 1.f;
    
    
    std::vector<float> fc_weight_scales = {post_scale_fc/0.00227380497381091f,
                                post_scale_fc/0.00276129716075956f,
                                post_scale_fc/0.0021945871412754f,
                                post_scale_fc/0.00212413328699767f,
                                post_scale_fc/0.0021686281543225f,
                                post_scale_fc/0.00226533785462379f,
                                post_scale_fc/0.00220444053411483f,
                                post_scale_fc/0.00189413060434162f,
                                post_scale_fc/0.00216874736361205f,
                                post_scale_fc/0.00201521744020283f,
                                post_scale_fc/0.00204910663887858f,
                                post_scale_fc/0.00237903092056512f,
                                post_scale_fc/0.00246293842792511f,
                                post_scale_fc/0.00250550871714949f,
                                post_scale_fc/0.0022863105405122f,
                                post_scale_fc/0.00180005235597491f,
                                post_scale_fc/0.00197727303020656f,
                                post_scale_fc/0.00226230616681277f,
                                post_scale_fc/0.0017212979728356f,
                                post_scale_fc/0.00265374663285911f,
                                post_scale_fc/0.00192045583389699f,
                                post_scale_fc/0.00205211713910102f,
                                post_scale_fc/0.00182974606286734f,
                                post_scale_fc/0.00202154810540378f,
                                post_scale_fc/0.00187521078623831f,
                                post_scale_fc/0.00184317701496183f,
                                post_scale_fc/0.00175985926762223f,
                                post_scale_fc/0.00204556202515959f,
                                post_scale_fc/0.00167550111655145f,
                                post_scale_fc/0.00208604405634105f,
                                post_scale_fc/0.00179914804175496f,
                                post_scale_fc/0.0017520097317174f,
                                post_scale_fc/0.00176897412165999f,
                                post_scale_fc/0.00188135122880339f,
                                post_scale_fc/0.00195123464800417f,
                                post_scale_fc/0.00196795188821852f,
                                post_scale_fc/0.00198700255714356f,
                                post_scale_fc/0.00185475393664091f,
                                post_scale_fc/0.00165263214148581f,
                                post_scale_fc/0.00172241579275578f,
                                post_scale_fc/0.00141988263931125f,
                                post_scale_fc/0.00176076241768896f,
                                post_scale_fc/0.00171943509485572f,
                                post_scale_fc/0.00171268451958894f,
                                post_scale_fc/0.00151844171341508f,
                                post_scale_fc/0.00187701731920242f,
                                post_scale_fc/0.00178259541280567f,
                                post_scale_fc/0.00205727014690637f,
                                post_scale_fc/0.00148176809307187f,
                                post_scale_fc/0.00182503531686961f,
                                post_scale_fc/0.0019412855617702f,
                                post_scale_fc/0.00202398817054927f,
                                post_scale_fc/0.00178466341458261f,
                                post_scale_fc/0.00197330070659518f,
                                post_scale_fc/0.00187695783097296f,
                                post_scale_fc/0.00194332562386989f,
                                post_scale_fc/0.00235910736955702f,
                                post_scale_fc/0.00227136840112507f,
                                post_scale_fc/0.00227734283544123f,
                                post_scale_fc/0.0015944829210639f,
                                post_scale_fc/0.00183581921737641f,
                                post_scale_fc/0.00243455148302018f,
                                post_scale_fc/0.00220791669562459f,
                                post_scale_fc/0.00191340572200715f,
                                post_scale_fc/0.00174400256946682f,
                                post_scale_fc/0.00203129136934876f,
                                post_scale_fc/0.00189180159941315f,
                                post_scale_fc/0.00187943095806986f,
                                post_scale_fc/0.00188917643390595f,
                                post_scale_fc/0.00241161580197513f,
                                post_scale_fc/0.00243983883410692f,
                                post_scale_fc/0.00179454823955893f,
                                post_scale_fc/0.0023817594628781f,
                                post_scale_fc/0.0026810101699084f,
                                post_scale_fc/0.0024981542956084f,
                                post_scale_fc/0.00206015328876674f,
                                post_scale_fc/0.00223715207539498f,
                                post_scale_fc/0.00233510066755116f,
                                post_scale_fc/0.00154552375897765f,
                                post_scale_fc/0.00182720355223864f,
                                post_scale_fc/0.00213516131043434f,
                                post_scale_fc/0.00164286722429096f,
                                post_scale_fc/0.00148240220732986f,
                                post_scale_fc/0.00247566308826208f,
                                post_scale_fc/0.00229545659385621f,
                                post_scale_fc/0.00176346860826015f,
                                post_scale_fc/0.00160409545060247f,
                                post_scale_fc/0.00245161750353872f,
                                post_scale_fc/0.00219946936704218f,
                                post_scale_fc/0.00170665839686989f,
                                post_scale_fc/0.00194122560787945f,
                                post_scale_fc/0.00192826520651578f,
                                post_scale_fc/0.00186766497790813f,
                                post_scale_fc/0.00184476177673786f,
                                post_scale_fc/0.00214149896055459f,
                                post_scale_fc/0.00197509303689003f,
                                post_scale_fc/0.00190958485472947f,
                                post_scale_fc/0.00170947448350489f,
                                post_scale_fc/0.00223785825073719f,
                                post_scale_fc/0.00140368286520242f,
                                post_scale_fc/0.0017365327803418f,
                                post_scale_fc/0.00235727499239146f,
                                post_scale_fc/0.00217136996798217f,
                                post_scale_fc/0.00184407876804471f,
                                post_scale_fc/0.00185165810398757f,
                                post_scale_fc/0.00310709001496434f,
                                post_scale_fc/0.00155488646123558f,
                                post_scale_fc/0.00235862960107624f,
                                post_scale_fc/0.00232213828712701f,
                                post_scale_fc/0.00214348337613046f,
                                post_scale_fc/0.00285129551775753f,
                                post_scale_fc/0.00259913643822073f,
                                post_scale_fc/0.00188678503036499f,
                                post_scale_fc/0.00202960358001291f,
                                post_scale_fc/0.00159880332648754f,
                                post_scale_fc/0.00368959014303982f,
                                post_scale_fc/0.0028271316550672f,
                                post_scale_fc/0.00248113158158957f,
                                post_scale_fc/0.00270411605015397f,
                                post_scale_fc/0.00226276786997914f,
                                post_scale_fc/0.00230042054317891f,
                                post_scale_fc/0.00259487703442573f,
                                post_scale_fc/0.00267121312208473f,
                                post_scale_fc/0.0021359717939049f,
                                post_scale_fc/0.00230317329987883f,
                                post_scale_fc/0.0018723786342889f,
                                post_scale_fc/0.00151497067417949f,
                                post_scale_fc/0.00160945684183388f,
                                post_scale_fc/0.00205510878004133f,
                                post_scale_fc/0.00208002096042037f,
                                post_scale_fc/0.00224806484766304f,
                                post_scale_fc/0.00191838189493864f,
                                post_scale_fc/0.00203480338677763f,
                                post_scale_fc/0.00169039471074938f,
                                post_scale_fc/0.00189313618466258f,
                                post_scale_fc/0.00173437292687594f,
                                post_scale_fc/0.00191510887816548f,
                                post_scale_fc/0.00181348028127104f,
                                post_scale_fc/0.00241745426319539f,
                                post_scale_fc/0.00140874262433499f,
                                post_scale_fc/0.00168895116075873f,
                                post_scale_fc/0.00190620205830782f,
                                post_scale_fc/0.00176185136660933f,
                                post_scale_fc/0.00155976612586528f,
                                post_scale_fc/0.00181939115282148f,
                                post_scale_fc/0.00175112159922719f,
                                post_scale_fc/0.00187609624117612f,
                                post_scale_fc/0.00160327181220054f,
                                post_scale_fc/0.00175550312269479f,
                                post_scale_fc/0.00220991251990199f,
                                post_scale_fc/0.00169767113402485f,
                                post_scale_fc/0.00191905698738992f,
                                post_scale_fc/0.00162297906354069f,
                                post_scale_fc/0.00178079027682542f,
                                post_scale_fc/0.00136614008806645f,
                                post_scale_fc/0.00157284981105476f,
                                post_scale_fc/0.00189219915773719f,
                                post_scale_fc/0.00185522926039993f,
                                post_scale_fc/0.00200293585658073f,
                                post_scale_fc/0.00144843710586428f,
                                post_scale_fc/0.00180994358379393f,
                                post_scale_fc/0.00171777443028986f,
                                post_scale_fc/0.00164927332662045f,
                                post_scale_fc/0.00159280956722795f,
                                post_scale_fc/0.00296406634151935f,
                                post_scale_fc/0.00150055170524865f,
                                post_scale_fc/0.00143491942435503f,
                                post_scale_fc/0.00135562592186033f,
                                post_scale_fc/0.00160850270185619f,
                                post_scale_fc/0.00175338389817625f,
                                post_scale_fc/0.00186567031778395f,
                                post_scale_fc/0.00179093040060251f,
                                post_scale_fc/0.00160077004693448f,
                                post_scale_fc/0.00161464768461883f,
                                post_scale_fc/0.00160517508629709f,
                                post_scale_fc/0.00142407929524779f,
                                post_scale_fc/0.00160076573956757f,
                                post_scale_fc/0.00174003967549651f,
                                post_scale_fc/0.00280921231023967f,
                                post_scale_fc/0.00182566954754292f,
                                post_scale_fc/0.00185128115117549f,
                                post_scale_fc/0.0017074286006391f,
                                post_scale_fc/0.00178550404962152f,
                                post_scale_fc/0.0022542888764292f,
                                post_scale_fc/0.00229900423437356f,
                                post_scale_fc/0.0018454446690157f,
                                post_scale_fc/0.00152663758490234f,
                                post_scale_fc/0.00165379280224442f,
                                post_scale_fc/0.00168062304146587f,
                                post_scale_fc/0.00194011931307613f,
                                post_scale_fc/0.00161389063578099f,
                                post_scale_fc/0.0019143606768921f,
                                post_scale_fc/0.00166649161837995f,
                                post_scale_fc/0.00160103966481983f,
                                post_scale_fc/0.00146010669413954f,
                                post_scale_fc/0.00145310978405177f,
                                post_scale_fc/0.00167770928237587f,
                                post_scale_fc/0.00142037658952176f,
                                post_scale_fc/0.00183380278758704f,
                                post_scale_fc/0.00184967962559312f,
                                post_scale_fc/0.00134795578196644f,
                                post_scale_fc/0.00160580035299062f,
                                post_scale_fc/0.00144138792529702f,
                                post_scale_fc/0.00178216863423585f,
                                post_scale_fc/0.0014596777036786f,
                                post_scale_fc/0.0015519387088716f,
                                post_scale_fc/0.00232749222777783f,
                                post_scale_fc/0.00178632594179362f,
                                post_scale_fc/0.00199006008915603f,
                                post_scale_fc/0.00211840355768799f,
                                post_scale_fc/0.00287890876643359f,
                                post_scale_fc/0.00166211277246475f,
                                post_scale_fc/0.00168912461958825f,
                                post_scale_fc/0.00178041774779558f,
                                post_scale_fc/0.00167344301007688f,
                                post_scale_fc/0.00158783781807869f,
                                post_scale_fc/0.00182641600258648f,
                                post_scale_fc/0.00204286398366093f,
                                post_scale_fc/0.00174577604047954f,
                                post_scale_fc/0.00197520037181675f,
                                post_scale_fc/0.00170922989491373f,
                                post_scale_fc/0.00191432028077542f,
                                post_scale_fc/0.0017963167047128f,
                                post_scale_fc/0.00180150067899376f,
                                post_scale_fc/0.00174359721131622f,
                                post_scale_fc/0.00184341345448046f,
                                post_scale_fc/0.00164247886277735f,
                                post_scale_fc/0.0018386070150882f,
                                post_scale_fc/0.00201458577066659f,
                                post_scale_fc/0.00213947403244674f,
                                post_scale_fc/0.00165432097855955f,
                                post_scale_fc/0.00172128656413406f,
                                post_scale_fc/0.00135096861049532f,
                                post_scale_fc/0.00193144485820084f,
                                post_scale_fc/0.00177033059298992f,
                                post_scale_fc/0.00148737418930977f,
                                post_scale_fc/0.00172317621763795f,
                                post_scale_fc/0.00206282804720103f,
                                post_scale_fc/0.00178454630076885f,
                                post_scale_fc/0.00174994149710983f,
                                post_scale_fc/0.00171817967202514f,
                                post_scale_fc/0.0017694963607937f,
                                post_scale_fc/0.00168888038024306f,
                                post_scale_fc/0.00153189909178763f,
                                post_scale_fc/0.00160436821170151f,
                                post_scale_fc/0.00182847899850457f,
                                post_scale_fc/0.00188812508713454f,
                                post_scale_fc/0.00145273783709853f,
                                post_scale_fc/0.00141829170752316f,
                                post_scale_fc/0.00188264262396842f,
                                post_scale_fc/0.00137655285652726f,
                                post_scale_fc/0.00195315689779818f,
                                post_scale_fc/0.00150350388139486f,
                                post_scale_fc/0.0018896113615483f,
                                post_scale_fc/0.00176461122464388f,
                                post_scale_fc/0.00174249673727899f,
                                post_scale_fc/0.00173693581018596f,
                                post_scale_fc/0.0013462376082316f,
                                post_scale_fc/0.00159559992607682f,
                                post_scale_fc/0.00167639984283596f,
                                post_scale_fc/0.00158776831813156f,
                                post_scale_fc/0.00178036466240882f,
                                post_scale_fc/0.00150549504905939f,
                                post_scale_fc/0.00144919601734727f,
                                post_scale_fc/0.00151322188321501f,
                                post_scale_fc/0.00182829028926789f,
                                post_scale_fc/0.00206678477115929f,
                                post_scale_fc/0.00163474597502499f,
                                post_scale_fc/0.00167773687280714f,
                                post_scale_fc/0.00174027553293854f,
                                post_scale_fc/0.00184615643229335f,
                                post_scale_fc/0.0021469322964549f,
                                post_scale_fc/0.00177943077869713f,
                                post_scale_fc/0.00207145139575004f,
                                post_scale_fc/0.00155303499195724f,
                                post_scale_fc/0.00160651456099003f,
                                post_scale_fc/0.00163724564481526f,
                                post_scale_fc/0.00149630079977214f,
                                post_scale_fc/0.00140780839137732f,
                                post_scale_fc/0.00173935480415821f,
                                post_scale_fc/0.00137331429868936f,
                                post_scale_fc/0.00226180627942085f,
                                post_scale_fc/0.00224515260197222f,
                                post_scale_fc/0.00162510038353502f,
                                post_scale_fc/0.00254597445018589f,
                                post_scale_fc/0.00253262720070779f,
                                post_scale_fc/0.00210423558019101f,
                                post_scale_fc/0.00260076345875859f,
                                post_scale_fc/0.00176997226662933f,
                                post_scale_fc/0.00192126911133527f,
                                post_scale_fc/0.00167852418962866f,
                                post_scale_fc/0.0019117840565741f,
                                post_scale_fc/0.00224045966751873f,
                                post_scale_fc/0.00184323848225176f,
                                post_scale_fc/0.00203229207545518f,
                                post_scale_fc/0.00183803762774914f,
                                post_scale_fc/0.0020669880323112f,
                                post_scale_fc/0.0018636651802808f,
                                post_scale_fc/0.00197544205002486f,
                                post_scale_fc/0.00182321318425238f,
                                post_scale_fc/0.0020713007543236f,
                                post_scale_fc/0.00196871580556035f,
                                post_scale_fc/0.0024854342918843f,
                                post_scale_fc/0.00181617983616888f,
                                post_scale_fc/0.0024674164596945f,
                                post_scale_fc/0.00256802490912377f,
                                post_scale_fc/0.00273384852334857f,
                                post_scale_fc/0.0020766204688698f,
                                post_scale_fc/0.00189530081115663f,
                                post_scale_fc/0.00211467873305082f,
                                post_scale_fc/0.00182136753574013f,
                                post_scale_fc/0.00135289737954735f,
                                post_scale_fc/0.00163317332044243f,
                                post_scale_fc/0.0020942660048604f,
                                post_scale_fc/0.00207786471582949f,
                                post_scale_fc/0.0022568495478481f,
                                post_scale_fc/0.00183024385478347f,
                                post_scale_fc/0.00239814189262688f,
                                post_scale_fc/0.00181208061985671f,
                                post_scale_fc/0.00190258619841188f,
                                post_scale_fc/0.00200146622955799f,
                                post_scale_fc/0.00193867180496454f,
                                post_scale_fc/0.00181898078881204f,
                                post_scale_fc/0.00156988471280783f,
                                post_scale_fc/0.00174589385278522f,
                                post_scale_fc/0.00201219739392399f,
                                post_scale_fc/0.00216762907803058f,
                                post_scale_fc/0.00191554112825542f,
                                post_scale_fc/0.0017467982834205f,
                                post_scale_fc/0.00218481151387095f,
                                post_scale_fc/0.00184446724597364f,
                                post_scale_fc/0.00162108184304088f,
                                post_scale_fc/0.0022005490027368f,
                                post_scale_fc/0.00185934151522815f,
                                post_scale_fc/0.00199351762421429f,
                                post_scale_fc/0.00194941449444741f,
                                post_scale_fc/0.00173150410410016f,
                                post_scale_fc/0.00182830321136862f,
                                post_scale_fc/0.00189346564002335f,
                                post_scale_fc/0.00208873930387198f,
                                post_scale_fc/0.00187532790005207f,
                                post_scale_fc/0.00235355924814939f,
                                post_scale_fc/0.00251186010427773f,
                                post_scale_fc/0.0025089019909501f,
                                post_scale_fc/0.00177288986742496f,
                                post_scale_fc/0.00234016170725226f,
                                post_scale_fc/0.00197643670253455f,
                                post_scale_fc/0.00223897281102836f,
                                post_scale_fc/0.00176799239125102f,
                                post_scale_fc/0.00188779167365282f,
                                post_scale_fc/0.00206255796365439f,
                                post_scale_fc/0.00247748964466154f,
                                post_scale_fc/0.00191615300718694f,
                                post_scale_fc/0.001792544266209f,
                                post_scale_fc/0.00227756751701235f,
                                post_scale_fc/0.00174876558594405f,
                                post_scale_fc/0.00160153943579643f,
                                post_scale_fc/0.00192522548604756f,
                                post_scale_fc/0.00166801002342253f,
                                post_scale_fc/0.001866843434982f,
                                post_scale_fc/0.00154662155546247f,
                                post_scale_fc/0.00205433135852217f,
                                post_scale_fc/0.0021028893534094f,
                                post_scale_fc/0.00240919645875692f,
                                post_scale_fc/0.0022338880226016f,
                                post_scale_fc/0.00188299675937742f,
                                post_scale_fc/0.00166554015595465f,
                                post_scale_fc/0.001706148032099f,
                                post_scale_fc/0.00164566549938172f,
                                post_scale_fc/0.0018590911058709f,
                                post_scale_fc/0.00183818640653043f,
                                post_scale_fc/0.00237788492813706f,
                                post_scale_fc/0.00191481073852628f,
                                post_scale_fc/0.0019383420003578f,
                                post_scale_fc/0.00189851108007133f,
                                post_scale_fc/0.00210291962139308f,
                                post_scale_fc/0.00160451175179332f,
                                post_scale_fc/0.00195236969739198f,
                                post_scale_fc/0.00167459750082343f,
                                post_scale_fc/0.00191429373808205f,
                                post_scale_fc/0.0020020492374897f,
                                post_scale_fc/0.00161326676607131f,
                                post_scale_fc/0.0018190547125414f,
                                post_scale_fc/0.00188410584814846f,
                                post_scale_fc/0.00203049881383776f,
                                post_scale_fc/0.00223667221143841f,
                                post_scale_fc/0.00209674099460244f,
                                post_scale_fc/0.0020359088666737f,
                                post_scale_fc/0.00201935623772442f,
                                post_scale_fc/0.00221077026799321f,
                                post_scale_fc/0.00190633547026664f,
                                post_scale_fc/0.00177474855445325f,
                                post_scale_fc/0.00185090128798037f,
                                post_scale_fc/0.00283018359914422f,
                                post_scale_fc/0.00172920268960297f,
                                post_scale_fc/0.0019179293885827f,
                                post_scale_fc/0.00236844504252076f,
                                post_scale_fc/0.00219054007902741f,
                                post_scale_fc/0.00267744669690728f,
                                post_scale_fc/0.00289597688242793f,
                                post_scale_fc/0.00202828808687627f,
                                post_scale_fc/0.00188326800707727f,
                                post_scale_fc/0.00263697584159672f,
                                post_scale_fc/0.0023885415866971f,
                                post_scale_fc/0.00226549617946147f,
                                post_scale_fc/0.00230370205827057f,
                                post_scale_fc/0.00198318460024893f,
                                post_scale_fc/0.00166168238501995f,
                                post_scale_fc/0.00190402055159211f,
                                post_scale_fc/0.00259186653420329f,
                                post_scale_fc/0.00238504470326006f,
                                post_scale_fc/0.00163115432951599f,
                                post_scale_fc/0.00148449407424777f,
                                post_scale_fc/0.00216616201214492f,
                                post_scale_fc/0.00227903365157544f,
                                post_scale_fc/0.00227849860675632f,
                                post_scale_fc/0.0033426065929234f,
                                post_scale_fc/0.00237518409267067f,
                                post_scale_fc/0.00177972484380006f,
                                post_scale_fc/0.00283443974331021f,
                                post_scale_fc/0.00276328460313379f,
                                post_scale_fc/0.0026895347982645f,
                                post_scale_fc/0.00249025621451437f,
                                post_scale_fc/0.00204224395565688f,
                                post_scale_fc/0.00232408149167895f,
                                post_scale_fc/0.00189173873513937f,
                                post_scale_fc/0.00172131787985563f,
                                post_scale_fc/0.00228119478560984f,
                                post_scale_fc/0.00179522600956261f,
                                post_scale_fc/0.00208577048033475f,
                                post_scale_fc/0.0029025103431195f,
                                post_scale_fc/0.00205257348716259f,
                                post_scale_fc/0.00207719043828547f,
                                post_scale_fc/0.00277605606243014f,
                                post_scale_fc/0.00293387100100517f,
                                post_scale_fc/0.00366692501120269f,
                                post_scale_fc/0.00154015549924224f,
                                post_scale_fc/0.00200198148377239f,
                                post_scale_fc/0.00188994174823164f,
                                post_scale_fc/0.0028432838153094f,
                                post_scale_fc/0.00247007794678211f,
                                post_scale_fc/0.00216810056008398f,
                                post_scale_fc/0.00179518514778465f,
                                post_scale_fc/0.00205323728732764f,
                                post_scale_fc/0.00247915578074753f,
                                post_scale_fc/0.00196093250997364f,
                                post_scale_fc/0.00227873236872255f,
                                post_scale_fc/0.00192023103591054f,
                                post_scale_fc/0.00225312123075127f,
                                post_scale_fc/0.00204433780163526f,
                                post_scale_fc/0.00279425154440104f,
                                post_scale_fc/0.0024464561138302f,
                                post_scale_fc/0.00212317844852805f,
                                post_scale_fc/0.00240832800045609f,
                                post_scale_fc/0.00260554510168731f,
                                post_scale_fc/0.00253293639980256f,
                                post_scale_fc/0.00280340481549501f,
                                post_scale_fc/0.00195568311028182f,
                                post_scale_fc/0.00212115771137177f,
                                post_scale_fc/0.00198116805404424f,
                                post_scale_fc/0.00166083662770688f,
                                post_scale_fc/0.00246666488237679f,
                                post_scale_fc/0.00272948970086872f,
                                post_scale_fc/0.00189584563486278f,
                                post_scale_fc/0.00195677042938768f,
                                post_scale_fc/0.00275812367908656f,
                                post_scale_fc/0.00254906364716589f,
                                post_scale_fc/0.00265903910622f,
                                post_scale_fc/0.00286629213951528f,
                                post_scale_fc/0.00210646190680563f,
                                post_scale_fc/0.00249965116381645f,
                                post_scale_fc/0.00207792199216783f,
                                post_scale_fc/0.00184900709427893f,
                                post_scale_fc/0.001968071796f,
                                post_scale_fc/0.0024408078752458f,
                                post_scale_fc/0.00205288594588637f,
                                post_scale_fc/0.00237410143017768f,
                                post_scale_fc/0.00171473308000713f,
                                post_scale_fc/0.00316213094629347f,
                                post_scale_fc/0.00145746092312037f,
                                post_scale_fc/0.0020093098282814f,
                                post_scale_fc/0.00201819255016744f,
                                post_scale_fc/0.00144999532494694f,
                                post_scale_fc/0.00198388146236538f,
                                post_scale_fc/0.00201412895694375f,
                                post_scale_fc/0.00159450469072908f,
                                post_scale_fc/0.00245110993273556f,
                                post_scale_fc/0.00225918693467974f,
                                post_scale_fc/0.00266961310990154f,
                                post_scale_fc/0.00393024086952209f,
                                post_scale_fc/0.00242827762849628f,
                                post_scale_fc/0.00247913133352994f,
                                post_scale_fc/0.00202380097471177f,
                                post_scale_fc/0.00231268815696239f,
                                post_scale_fc/0.00219374289736151f,
                                post_scale_fc/0.00220095668919384f,
                                post_scale_fc/0.00186145829502493f,
                                post_scale_fc/0.00171963672619313f,
                                post_scale_fc/0.00222481857053935f,
                                post_scale_fc/0.00197305111214518f,
                                post_scale_fc/0.0022710426710546f,
                                post_scale_fc/0.00220544007606804f,
                                post_scale_fc/0.00227675097994506f,
                                post_scale_fc/0.0024713312741369f,
                                post_scale_fc/0.00210204371251165f,
                                post_scale_fc/0.00228276429697871f,
                                post_scale_fc/0.00221834750846028f,
                                post_scale_fc/0.00189642247278243f,
                                post_scale_fc/0.00300059700384736f,
                                post_scale_fc/0.0023351521231234f,
                                post_scale_fc/0.00223834556527435f,
                                post_scale_fc/0.00185367837548255f,
                                post_scale_fc/0.00169567111879587f,
                                post_scale_fc/0.0020282263867557f,
                                post_scale_fc/0.00201443210244178f,
                                post_scale_fc/0.00321284099481999f,
                                post_scale_fc/0.00213504256680607f,
                                post_scale_fc/0.001592404441908f,
                                post_scale_fc/0.00204032124020159f,
                                post_scale_fc/0.00249219802208244f,
                                post_scale_fc/0.00209611933678388f,
                                post_scale_fc/0.00215063290670514f,
                                post_scale_fc/0.0019387659849599f,
                                post_scale_fc/0.00265639857389032f,
                                post_scale_fc/0.0021613985300064f,
                                post_scale_fc/0.00212164735421538f,
                                post_scale_fc/0.00187859579455107f,
                                post_scale_fc/0.00175356667023152f,
                                post_scale_fc/0.00169870466925203f,
                                post_scale_fc/0.00206679943948984f,
                                post_scale_fc/0.00198687007650733f,
                                post_scale_fc/0.00164531241171062f,
                                post_scale_fc/0.00233347225002944f,
                                post_scale_fc/0.00363705982454121f,
                                post_scale_fc/0.00245634349994361f,
                                post_scale_fc/0.00182601914275437f,
                                post_scale_fc/0.0019087390974164f,
                                post_scale_fc/0.00237506907433271f,
                                post_scale_fc/0.00237695127725601f,
                                post_scale_fc/0.00275911041535437f,
                                post_scale_fc/0.00255021639168262f,
                                post_scale_fc/0.00179510866291821f,
                                post_scale_fc/0.00227887416258454f,
                                post_scale_fc/0.00263619888573884f,
                                post_scale_fc/0.00259410100989043f,
                                post_scale_fc/0.00218834658153355f,
                                post_scale_fc/0.00266179163008928f,
                                post_scale_fc/0.00290350569412112f,
                                post_scale_fc/0.00248509785160422f,
                                post_scale_fc/0.00182168814353644f,
                                post_scale_fc/0.00186321034561842f,
                                post_scale_fc/0.00224497052840888f,
                                post_scale_fc/0.00217848294414579f,
                                post_scale_fc/0.00228785583749413f,
                                post_scale_fc/0.00258980644866824f,
                                post_scale_fc/0.00232384703122079f,
                                post_scale_fc/0.00304048205725848f,
                                post_scale_fc/0.00217924616299569f,
                                post_scale_fc/0.00193749321624636f,
                                post_scale_fc/0.00229281652718782f,
                                post_scale_fc/0.00208604033105075f,
                                post_scale_fc/0.00212690141052007f,
                                post_scale_fc/0.00314504886046052f,
                                post_scale_fc/0.00186429254245013f,
                                post_scale_fc/0.00255918106995522f,
                                post_scale_fc/0.00290748104453086f,
                                post_scale_fc/0.00203401013277471f,
                                post_scale_fc/0.00323416967876255f,
                                post_scale_fc/0.00210816576145589f,
                                post_scale_fc/0.0023302671033889f,
                                post_scale_fc/0.0024798687081784f,
                                post_scale_fc/0.00178731279447674f,
                                post_scale_fc/0.0017893366748467f,
                                post_scale_fc/0.00248434394598007f,
                                post_scale_fc/0.00182808574754744f,
                                post_scale_fc/0.00237280852161347f,
                                post_scale_fc/0.00240111444145441f,
                                post_scale_fc/0.00203824695199728f,
                                post_scale_fc/0.00249841064214706f,
                                post_scale_fc/0.0019625558052212f,
                                post_scale_fc/0.00318012828938663f,
                                post_scale_fc/0.0022335909307003f,
                                post_scale_fc/0.00171474053058773f,
                                post_scale_fc/0.00216643628664314f,
                                post_scale_fc/0.00155677984002977f,
                                post_scale_fc/0.00177534925751388f,
                                post_scale_fc/0.00183719117194414f,
                                post_scale_fc/0.00175246503204107f,
                                post_scale_fc/0.0044563109986484f,
                                post_scale_fc/0.00176297465804964f,
                                post_scale_fc/0.00165702716913074f,
                                post_scale_fc/0.00248482776805758f,
                                post_scale_fc/0.00235392758622765f,
                                post_scale_fc/0.00225751288235187f,
                                post_scale_fc/0.00282271648757159f,
                                post_scale_fc/0.00209256866946816f,
                                post_scale_fc/0.00234413985162973f,
                                post_scale_fc/0.00150976574514061f,
                                post_scale_fc/0.00259015429764986f,
                                post_scale_fc/0.00330797955393791f,
                                post_scale_fc/0.00135345535818487f,
                                post_scale_fc/0.00240114866755902f,
                                post_scale_fc/0.00318259629420936f,
                                post_scale_fc/0.00198164372704923f,
                                post_scale_fc/0.00215690047480165f,
                                post_scale_fc/0.00179972907062619f,
                                post_scale_fc/0.00275643775239586f,
                                post_scale_fc/0.0020660338923335f,
                                post_scale_fc/0.00578146800398826f,
                                post_scale_fc/0.00183471769560128f,
                                post_scale_fc/0.00238805613480508f,
                                post_scale_fc/0.00417285226285457f,
                                post_scale_fc/0.00210732594132423f,
                                post_scale_fc/0.00171622540801763f,
                                post_scale_fc/0.00191407464444637f,
                                post_scale_fc/0.00212195911444723f,
                                post_scale_fc/0.0022712699137628f,
                                post_scale_fc/0.00273464573547244f,
                                post_scale_fc/0.00169114384334534f,
                                post_scale_fc/0.00186147761996835f,
                                post_scale_fc/0.00224705645814538f,
                                post_scale_fc/0.00184636446647346f,
                                post_scale_fc/0.00201024510897696f,
                                post_scale_fc/0.00200690561905503f,
                                post_scale_fc/0.00228176754899323f,
                                post_scale_fc/0.00315204681828618f,
                                post_scale_fc/0.0019847119692713f,
                                post_scale_fc/0.00226551620289683f,
                                post_scale_fc/0.00217244122177362f,
                                post_scale_fc/0.00223000883124768f,
                                post_scale_fc/0.00224500452168285f,
                                post_scale_fc/0.00212041684426367f,
                                post_scale_fc/0.00193466816563159f,
                                post_scale_fc/0.00166088237892836f,
                                post_scale_fc/0.00271178549155592f,
                                post_scale_fc/0.00171118765138089f,
                                post_scale_fc/0.00151388684753328f,
                                post_scale_fc/0.00178832898382097f,
                                post_scale_fc/0.00166956265456974f,
                                post_scale_fc/0.00199061236344277f,
                                post_scale_fc/0.00241334992460906f,
                                post_scale_fc/0.00229916465468704f,
                                post_scale_fc/0.00279636238701641f,
                                post_scale_fc/0.0022711744531989f,
                                post_scale_fc/0.00194922881200909f,
                                post_scale_fc/0.00232420791871845f,
                                post_scale_fc/0.0028739902190864f,
                                post_scale_fc/0.00211476488038897f,
                                post_scale_fc/0.00225583021529018f,
                                post_scale_fc/0.0016526662511751f,
                                post_scale_fc/0.00194803986232727f,
                                post_scale_fc/0.0023809727281332f,
                                post_scale_fc/0.00292756641283631f,
                                post_scale_fc/0.00200345669873058f,
                                post_scale_fc/0.00169874809216707f,
                                post_scale_fc/0.00178563233930617f,
                                post_scale_fc/0.00197020941413939f,
                                post_scale_fc/0.00208938517607748f,
                                post_scale_fc/0.00253145559690892f,
                                post_scale_fc/0.00234960881061851f,
                                post_scale_fc/0.0018707423005253f,
                                post_scale_fc/0.00157278426922857f,
                                post_scale_fc/0.00193510355893522f,
                                post_scale_fc/0.0017722196644172f,
                                post_scale_fc/0.00232567405328154f,
                                post_scale_fc/0.00212915032170712f,
                                post_scale_fc/0.00163731584325432f,
                                post_scale_fc/0.00241859117522835f,
                                post_scale_fc/0.0021908467169851f,
                                post_scale_fc/0.00260530738160014f,
                                post_scale_fc/0.00217398628592491f,
                                post_scale_fc/0.00190005078911781f,
                                post_scale_fc/0.00242758193053305f,
                                post_scale_fc/0.00165101792663335f,
                                post_scale_fc/0.00185853487346321f,
                                post_scale_fc/0.00176426384132355f,
                                post_scale_fc/0.0018990309908986f,
                                post_scale_fc/0.00209580897353589f,
                                post_scale_fc/0.00224685785360634f,
                                post_scale_fc/0.00275767501443624f,
                                post_scale_fc/0.00164664664771407f,
                                post_scale_fc/0.00180674367584288f,
                                post_scale_fc/0.00227510510012507f,
                                post_scale_fc/0.00201213755644857f,
                                post_scale_fc/0.00344277429394423f,
                                post_scale_fc/0.00207265163771808f,
                                post_scale_fc/0.0017688653897494f,
                                post_scale_fc/0.00195470172911882f,
                                post_scale_fc/0.00176204915624111f,
                                post_scale_fc/0.0019983050879091f,
                                post_scale_fc/0.00205031572841107f,
                                post_scale_fc/0.00159256823826581f,
                                post_scale_fc/0.00165684113744646f,
                                post_scale_fc/0.00164969952311366f,
                                post_scale_fc/0.00200045900419354f,
                                post_scale_fc/0.00197013933211565f,
                                post_scale_fc/0.0022674836218357f,
                                post_scale_fc/0.00218598544597625f,
                                post_scale_fc/0.00226586312055587f,
                                post_scale_fc/0.00206865789368748f,
                                post_scale_fc/0.00356585718691349f,
                                post_scale_fc/0.0019755105022341f,
                                post_scale_fc/0.00311614526435732f,
                                post_scale_fc/0.00216384744271636f,
                                post_scale_fc/0.00183811620809137f,
                                post_scale_fc/0.00213912362232804f,
                                post_scale_fc/0.00224972190335392f,
                                post_scale_fc/0.00195866567082703f,
                                post_scale_fc/0.00211755628697574f,
                                post_scale_fc/0.00159794988576322f,
                                post_scale_fc/0.0023316410370171f,
                                post_scale_fc/0.0022115169558674f,
                                post_scale_fc/0.00272747152484953f,
                                post_scale_fc/0.00181745493318885f,
                                post_scale_fc/0.00182366208173334f,
                                post_scale_fc/0.00226789130829274f,
                                post_scale_fc/0.00338768912479281f,
                                post_scale_fc/0.00171551981475204f,
                                post_scale_fc/0.0019521452486515f,
                                post_scale_fc/0.00227167154662311f,
                                post_scale_fc/0.00211142562329769f,
                                post_scale_fc/0.00252817128784954f,
                                post_scale_fc/0.00276511022821068f,
                                post_scale_fc/0.00205451226793229f,
                                post_scale_fc/0.00261727953329682f,
                                post_scale_fc/0.00206659850664436f,
                                post_scale_fc/0.00196980405598878f,
                                post_scale_fc/0.00179315079003572f,
                                post_scale_fc/0.0033456867095083f,
                                post_scale_fc/0.00192409346345812f,
                                post_scale_fc/0.00234094192273914f,
                                post_scale_fc/0.00169258879031986f,
                                post_scale_fc/0.00167121470440179f,
                                post_scale_fc/0.00198900722898542f,
                                post_scale_fc/0.00185921334195882f,
                                post_scale_fc/0.00260688201524317f,
                                post_scale_fc/0.00243259849958121f,
                                post_scale_fc/0.00199602195061743f,
                                post_scale_fc/0.00225037662312388f,
                                post_scale_fc/0.00209766114130616f,
                                post_scale_fc/0.00228457897901535f,
                                post_scale_fc/0.00259153265506029f,
                                post_scale_fc/0.00185632903594523f,
                                post_scale_fc/0.00190604210365563f,
                                post_scale_fc/0.00202557048760354f,
                                post_scale_fc/0.00191448011901229f,
                                post_scale_fc/0.0023406189866364f,
                                post_scale_fc/0.0019396828720346f,
                                post_scale_fc/0.00192890781909227f,
                                post_scale_fc/0.00210989918559789f,
                                post_scale_fc/0.00204224348999559f,
                                post_scale_fc/0.0019594389013946f,
                                post_scale_fc/0.00209293374791741f,
                                post_scale_fc/0.00203690282069146f,
                                post_scale_fc/0.0015879925340414f,
                                post_scale_fc/0.00184804352466017f,
                                post_scale_fc/0.00191176624502986f,
                                post_scale_fc/0.00163238262757658f,
                                post_scale_fc/0.00193511508405208f,
                                post_scale_fc/0.00166031729895621f,
                                post_scale_fc/0.00361558841541409f,
                                post_scale_fc/0.00167956762015819f,
                                post_scale_fc/0.0021192783024162f,
                                post_scale_fc/0.00202689063735306f,
                                post_scale_fc/0.00184074568096548f,
                                post_scale_fc/0.00236637494526803f,
                                post_scale_fc/0.00233529438264668f,
                                post_scale_fc/0.00184578949119895f,
                                post_scale_fc/0.00231261225417256f,
                                post_scale_fc/0.00289369304664433f,
                                post_scale_fc/0.00268775061704218f,
                                post_scale_fc/0.00155028363224118f,
                                post_scale_fc/0.00218705320730805f,
                                post_scale_fc/0.00242934864945709f,
                                post_scale_fc/0.00191826466470956f,
                                post_scale_fc/0.00211014971137046f,
                                post_scale_fc/0.00246789562515914f,
                                post_scale_fc/0.00204663653858006f,
                                post_scale_fc/0.00198838277719914f,
                                post_scale_fc/0.00463508768007159f,
                                post_scale_fc/0.00243761297315359f,
                                post_scale_fc/0.00194274657405912f,
                                post_scale_fc/0.00303290761075913f,
                                post_scale_fc/0.00195377459749579f,
                                post_scale_fc/0.00243469537235796f,
                                post_scale_fc/0.00264217425137758f,
                                post_scale_fc/0.00266873906366527f,
                                post_scale_fc/0.00236693187616765f,
                                post_scale_fc/0.00240744464099407f,
                                post_scale_fc/0.00301293074153363f,
                                post_scale_fc/0.00391360279172658f,
                                post_scale_fc/0.002145417034626f,
                                post_scale_fc/0.00192571757361292f,
                                post_scale_fc/0.00269744708202779f,
                                post_scale_fc/0.00315934419631958f,
                                post_scale_fc/0.00224426575005054f,
                                post_scale_fc/0.0018723455723375f,
                                post_scale_fc/0.00238603446632623f,
                                post_scale_fc/0.00214536441490054f,
                                post_scale_fc/0.00181536190211772f,
                                post_scale_fc/0.00205451250076293f,
                                post_scale_fc/0.00212743086740374f,
                                post_scale_fc/0.00256292126141488f,
                                post_scale_fc/0.00251797819510102f,
                                post_scale_fc/0.0014911942416802f,
                                post_scale_fc/0.00201508798636496f,
                                post_scale_fc/0.00234746350906789f,
                                post_scale_fc/0.00194500351790338f,
                                post_scale_fc/0.00282566272653639f,
                                post_scale_fc/0.00176103378180414f,
                                post_scale_fc/0.00339953345246613f,
                                post_scale_fc/0.00181273382622748f,
                                post_scale_fc/0.00212915148586034f,
                                post_scale_fc/0.00183940900024026f,
                                post_scale_fc/0.00224190764129161f,
                                post_scale_fc/0.00285854353569448f,
                                post_scale_fc/0.00231267698109149f,
                                post_scale_fc/0.00242159469053149f,
                                post_scale_fc/0.00226553226821124f,
                                post_scale_fc/0.00191547232680022f,
                                post_scale_fc/0.00205241818912327f,
                                post_scale_fc/0.00175638997461646f,
                                post_scale_fc/0.00270381942391395f,
                                post_scale_fc/0.0018859093543142f,
                                post_scale_fc/0.00268288701772689f,
                                post_scale_fc/0.00234548584558069f,
                                post_scale_fc/0.00191822578199207f,
                                post_scale_fc/0.00186813226900994f,
                                post_scale_fc/0.0027695894241333f,
                                post_scale_fc/0.00176504394039511f,
                                post_scale_fc/0.00163183535914868f,
                                post_scale_fc/0.00290826358832418f,
                                post_scale_fc/0.00218442408367991f,
                                post_scale_fc/0.00213779578916728f,
                                post_scale_fc/0.00331912096589803f,
                                post_scale_fc/0.00171584880445152f,
                                post_scale_fc/0.00383840571157634f,
                                post_scale_fc/0.00384032214060425f,
                                post_scale_fc/0.00210095639340579f,
                                post_scale_fc/0.00191184412688016f,
                                post_scale_fc/0.00212015444412827f,
                                post_scale_fc/0.00187981675844639f,
                                post_scale_fc/0.00268462533131241f,
                                post_scale_fc/0.00315822893753647f,
                                post_scale_fc/0.00180805369745939f,
                                post_scale_fc/0.00193846225738525f,
                                post_scale_fc/0.00203489046543836f,
                                post_scale_fc/0.00198656297288835f,
                                post_scale_fc/0.00153718155343085f,
                                post_scale_fc/0.00202228967100381f,
                                post_scale_fc/0.00169703492429107f,
                                post_scale_fc/0.00394117645919323f,
                                post_scale_fc/0.00317592383362352f,
                                post_scale_fc/0.00328714144416153f,
                                post_scale_fc/0.00333709572441875f,
                                post_scale_fc/0.00214883754961192f,
                                post_scale_fc/0.00242167944088578f,
                                post_scale_fc/0.00190275255590677f,
                                post_scale_fc/0.00466032279655337f,
                                post_scale_fc/0.00267692445777356f,
                                post_scale_fc/0.00253167888149619f,
                                post_scale_fc/0.00228986656293272f,
                                post_scale_fc/0.00329640228301286f,
                                post_scale_fc/0.00320884934626519f,
                                post_scale_fc/0.0020452591124922f,
                                post_scale_fc/0.00246208626776933f,
                                post_scale_fc/0.00224645435810089f,
                                post_scale_fc/0.00223849271424114f,
                                post_scale_fc/0.00200690934434533f,
                                post_scale_fc/0.00174108438659459f,
                                post_scale_fc/0.00185273552779108f,
                                post_scale_fc/0.00240103318355977f,
                                post_scale_fc/0.00190464686602354f,
                                post_scale_fc/0.0019472079584375f,
                                post_scale_fc/0.0021951210219413f,
                                post_scale_fc/0.00297502661123871f,
                                post_scale_fc/0.00339040951803326f,
                                post_scale_fc/0.00247981632128357f,
                                post_scale_fc/0.00362138031050562f,
                                post_scale_fc/0.0022784189786762f,
                                post_scale_fc/0.00198276154696941f,
                                post_scale_fc/0.00249466323293745f,
                                post_scale_fc/0.00162394437938928f,
                                post_scale_fc/0.00193656235933303f,
                                post_scale_fc/0.00192566134501248f,
                                post_scale_fc/0.0038460623472929f,
                                post_scale_fc/0.00183368159923702f,
                                post_scale_fc/0.00179789715912193f,
                                post_scale_fc/0.00203241547569632f,
                                post_scale_fc/0.00228171772323548f,
                                post_scale_fc/0.00225571822375059f,
                                post_scale_fc/0.00259172613732516f,
                                post_scale_fc/0.00229148473590612f,
                                post_scale_fc/0.00164745410438627f,
                                post_scale_fc/0.00245929043740034f,
                                post_scale_fc/0.00241900375112891f,
                                post_scale_fc/0.00279543595388531f,
                                post_scale_fc/0.00270285760052502f,
                                post_scale_fc/0.00168569420929998f,
                                post_scale_fc/0.00151216506492346f,
                                post_scale_fc/0.00219536223448812f,
                                post_scale_fc/0.00200520176440477f,
                                post_scale_fc/0.00178698345553129f,
                                post_scale_fc/0.00326062832027673f,
                                post_scale_fc/0.00242430088110268f,
                                post_scale_fc/0.00291381543502211f,
                                post_scale_fc/0.00225287186913192f,
                                post_scale_fc/0.00287016900256276f,
                                post_scale_fc/0.00232223095372319f,
                                post_scale_fc/0.0022382098250091f,
                                post_scale_fc/0.00246563018299639f,
                                post_scale_fc/0.00245226360857486f,
                                post_scale_fc/0.00313904695212841f,
                                post_scale_fc/0.00248649273999035f,
                                post_scale_fc/0.00217115832492709f,
                                post_scale_fc/0.00252142688259482f,
                                post_scale_fc/0.00312265125103294f,
                                post_scale_fc/0.00224291929043829f,
                                post_scale_fc/0.00195017899386584f,
                                post_scale_fc/0.00176159490365535f,
                                post_scale_fc/0.00222711148671805f,
                                post_scale_fc/0.00250628544017672f,
                                post_scale_fc/0.00251064309850335f,
                                post_scale_fc/0.00189737894106656f,
                                post_scale_fc/0.00156293658073991f,
                                post_scale_fc/0.00271065277047455f,
                                post_scale_fc/0.00177337031345814f,
                                post_scale_fc/0.00177155272103846f,
                                post_scale_fc/0.0019618400838226f,
                                post_scale_fc/0.0020773340947926f,
                                post_scale_fc/0.00222243182361125f,
                                post_scale_fc/0.00189561047591269f,
                                post_scale_fc/0.00262672360986471f,
                                post_scale_fc/0.00207616342231631f,
                                post_scale_fc/0.00186561339069157f,
                                post_scale_fc/0.00184982572682201f,
                                post_scale_fc/0.00269384356215596f,
                                post_scale_fc/0.00280993711203336f,
                                post_scale_fc/0.00247844075784087f,
                                post_scale_fc/0.0024635512381792f,
                                post_scale_fc/0.00222483836114406f,
                                post_scale_fc/0.00258312909863889f,
                                post_scale_fc/0.00246577127836644f,
                                post_scale_fc/0.00222072377800941f,
                                post_scale_fc/0.00164023239631205f,
                                post_scale_fc/0.00186130951624363f,
                                post_scale_fc/0.00213983166031539f,
                                post_scale_fc/0.00209297589026391f,
                                post_scale_fc/0.00257288943976163f,
                                post_scale_fc/0.00222503044642508f,
                                post_scale_fc/0.00337271019816398f,
                                post_scale_fc/0.00300134485587477f,
                                post_scale_fc/0.00238478020764887f,
                                post_scale_fc/0.00265213404782116f,
                                post_scale_fc/0.00347232981584966f,
                                post_scale_fc/0.00235541886650025f,
                                post_scale_fc/0.00225467793643474f,
                                post_scale_fc/0.0027637593448162f,
                                post_scale_fc/0.00238223164342343f,
                                post_scale_fc/0.00157080520875751f,
                                post_scale_fc/0.00327736581675708f,
                                post_scale_fc/0.00232435879297554f,
                                post_scale_fc/0.0022538264747709f,
                                post_scale_fc/0.00176513579208403f,
                                post_scale_fc/0.00208830437622964f,
                                post_scale_fc/0.00358790764585137f,
                                post_scale_fc/0.00240409513935446f,
                                post_scale_fc/0.00168346124701201f,
                                post_scale_fc/0.00162452540826052f,
                                post_scale_fc/0.00224283430725336f,
                                post_scale_fc/0.00242280843667686f,
                                post_scale_fc/0.002370415488258f,
                                post_scale_fc/0.00245956308208405f,
                                post_scale_fc/0.00234555639326572f,
                                post_scale_fc/0.00283148512244224f,
                                post_scale_fc/0.00174820038955658f,
                                post_scale_fc/0.00249198195524513f,
                                post_scale_fc/0.00210990943014621f,
                                post_scale_fc/0.0019511777209118f,
                                post_scale_fc/0.00188935827463865f,
                                post_scale_fc/0.00192921736743301f,
                                post_scale_fc/0.00244033988565206f,
                                post_scale_fc/0.00222765794023871f,
                                post_scale_fc/0.00204103579744696f,
                                post_scale_fc/0.00275558186694979f,
                                post_scale_fc/0.00204080156981945f,
                                post_scale_fc/0.00231120688840746f,
                                post_scale_fc/0.00246676709502935f,
                                post_scale_fc/0.00211023981682956f,
                                post_scale_fc/0.00234567653387784f,
                                post_scale_fc/0.00192438776139169f,
                                post_scale_fc/0.00236339983530342f,
                                post_scale_fc/0.00236061634495854f,
                                post_scale_fc/0.00221207714639604f,
                                post_scale_fc/0.00232370081357657f,
                                post_scale_fc/0.00274417200125753f,
                                post_scale_fc/0.00257563823834061f,
                                post_scale_fc/0.00196836469694972f,
                                post_scale_fc/0.00294900848530232f,
                                post_scale_fc/0.00322553888f};
    
    
    
    
    
    

    const float in_scale_fc = 0.18475806713104248;

    std::vector<float> fc_scales(1000);
    std::vector<float> fc_bias_scales(1000);


    for(int i=0;i<1000;i++)
    {
        fc_scales[i] = in_scale_fc*(1/fc_weight_scales[i]);
        
    }
    for(int j=0;j<1000;j++)
    {
        fc_bias_scales[j] = 1/fc_scales[j];
    }
    
  
    
    auto matmul_d = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference,fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md);

    dnnl::primitive_attr fc_dst_attr;
    fc_dst_attr.set_output_scales(2, fc_scales);
    fc_dst_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
   


    fc_bias_memory = dnnl::memory({fc_bias_tz_, dt::f32, tag::a}, eng_dnn_);
    dnnl::primitive_attr fc_bias_attr;
    fc_bias_attr.set_output_scales(fc_bias_mask,fc_bias_scales);
    auto fc_bias_reorder_pd = dnnl::reorder::primitive_desc(eng_dnn_, fc_user_bias_memory.get_desc(),
                                                         eng_dnn_, fc_bias_memory.get_desc(), fc_bias_attr);
    auto fc_bias_reorder = dnnl::reorder(fc_bias_reorder_pd);
    fc_bias_reorder.execute(dnn_strm_, fc_user_bias_memory, fc_bias_memory);

   
    auto fc_prim_desc = dnnl::inner_product_forward::primitive_desc(matmul_d, fc_dst_attr,eng_dnn_);
    post_scratchpad_md_prim_ = fc_prim_desc.scratchpad_desc();
    fc_weights_memory = dnnl::memory(fc_prim_desc.weights_desc(), eng_dnn_);

    dnnl::primitive_attr fc_weight_attr;
    fc_weight_attr.set_output_scales(fc_weight_mask,fc_weight_scales);
    auto fc_weight_reorder_pd = dnnl::reorder::primitive_desc(eng_dnn_, fc_user_weights_memory.get_desc(),
                                                           eng_dnn_, fc_weights_memory.get_desc(), fc_weight_attr);
    auto fc_weight_reorder = dnnl::reorder(fc_weight_reorder_pd);
    fc_weight_reorder.execute(dnn_strm_, fc_user_weights_memory, fc_weights_memory);

    fc_forward_prim_ = dnnl::inner_product_forward(fc_prim_desc);
    memcpy(fc_weights_ptr_end_, fc_weights_memory.get_data_handle(), fc_weights_memory.get_desc().get_size());
    memcpy(fc_bias_ptr_end_, fc_bias_memory.get_data_handle(), 1000*sizeof(float));

    init_onednn = true;
}

static void runEnd(int8_t* middle_out, int8_t* avg_pool_output, float* final_out, void* fc_scratch_ptr, int8_t* fc_src_mem)
{

  dnnl::memory avg_pool_src_memory_ = dnnl::memory({{avgpool_src_tz_}, dt::s8, tag::nhwc}, eng_dnn_,middle_out);
  dnnl::memory avg_pool_dst_memory_ = dnnl::memory({{avgpool_dst_tz_}, dt::s8, tag::nhwc}, eng_dnn_,avg_pool_output);
  dnnl::memory fc_src_memory_ = dnnl::memory({{fc_src_tz_}, dt::s8, tag::nhwc}, eng_dnn_,fc_src_mem);
  dnnl::memory fc_dst_memory_ = dnnl::memory({{fc_dst_tz_}, dt::f32, tag::nc}, eng_dnn_,final_out);
  dnnl::memory fc_scratch_memory = dnnl::memory(post_scratchpad_md_prim_, eng_dnn_, fc_scratch_ptr);

  avg_pool_forward_prim_.execute(dnn_strm_, {{DNNL_ARG_SRC, avg_pool_src_memory_},
                         {DNNL_ARG_DST, avg_pool_dst_memory_}});

  fc_input_reorder.execute(dnn_strm_,{{DNNL_ARG_SRC, avg_pool_dst_memory_},
                            {DNNL_ARG_DST, fc_src_memory_}});

  fc_forward_prim_.execute(dnn_strm_, {{DNNL_ARG_SRC, fc_src_memory_},
                        {DNNL_ARG_WEIGHTS, fc_weights_memory},
                        {DNNL_ARG_BIAS, fc_bias_memory},
                        {DNNL_ARG_DST, fc_dst_memory_},
                        {DNNL_ARG_SCRATCHPAD, fc_scratch_memory}});

}
static void runStart(int8_t* input_pointer, int8_t* conv_out_pointer, int8_t* output_pointer, void* conv_scratch_ptr) {
  dnnl::memory conv_src_memory = dnnl::memory({{conv_src_tz_stg1_}, dt::s8, tag::nhwc}, eng_dnn_, input_pointer);
  dnnl::memory conv_dst_memory = dnnl::memory({{conv_dst_tz_stg1_}, dt::s8, tag::nhwc}, eng_dnn_, conv_out_pointer);
  dnnl::memory pool_dst_memory = dnnl::memory({{maxpool_dst_tz_}, dt::s8, tag::nhwc}, eng_dnn_, output_pointer);
  dnnl::memory conv_scratch_memory = dnnl::memory(scratchpad_md_prim_, eng_dnn_, conv_scratch_ptr);

  conv_forward_prim_.execute(dnn_strm_, {{DNNL_ARG_SRC, conv_src_memory},
                                 {DNNL_ARG_WEIGHTS, conv_weights_memory},
                                 {DNNL_ARG_BIAS, conv_bias_memory},
                                 {DNNL_ARG_DST, conv_dst_memory},
                                 {DNNL_ARG_SCRATCHPAD, conv_scratch_memory}});

  pool_forward_prim_.execute(dnn_strm_, {{DNNL_ARG_SRC, conv_dst_memory},
                                 {DNNL_ARG_DST, pool_dst_memory}});
}

extern "C" void sc_init_rn50_backbone_bs8();
extern "C" void sc_init_rn50_backbone_wrapper_bs8(float* conv1_weight, float* conv1_bias, float* fc_weight, float* fc_bias) {
  prepareOneDNN(conv1_weight, conv1_bias,fc_weight,fc_bias);
  sc_init_rn50_backbone_bs8();
}

static bool mlperf_batchwise_8_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_reorder_res2a_conv_2_cast_mul_add_cast_add_cast_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_res2c_conv_0_cast_mul_add_relu_cast_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_reorder_res3d_conv_2_cast_mul_add_cast_add_cast__685(int8_t* __restrict__ __outs_0, int64_t* __restrict__ input_pointers, int8_t* __restrict__ __ins_1, float* __restrict__ __ins_2, float* __restrict__ __ins_3, int8_t* __restrict__ __ins_4, float* __restrict__ __ins_5, float* __restrict__ __ins_6, int8_t* __restrict__ __ins_7, float* __restrict__ __ins_8, float* __restrict__ __ins_9, int8_t* __restrict__ __ins_10, float* __restrict__ __ins_11, float* __restrict__ __ins_12, int8_t* __restrict__ __ins_13, float* __restrict__ __ins_14, float* __restrict__ __ins_15, int8_t* __restrict__ __ins_16, float* __restrict__ __ins_17, float* __restrict__ __ins_18, int8_t* __restrict__ __ins_19, float* __restrict__ __ins_20, float* __restrict__ __ins_21, int8_t* __restrict__ __ins_22, float* __restrict__ __ins_23, float* __restrict__ __ins_24, int8_t* __restrict__ __ins_25, float* __restrict__ __ins_26, float* __restrict__ __ins_27, int8_t* __restrict__ __ins_28, float* __restrict__ __ins_29, float* __restrict__ __ins_30, int8_t* __restrict__ __ins_31, float* __restrict__ __ins_32, float* __restrict__ __ins_33, int8_t* __restrict__ __ins_34, float* __restrict__ __ins_35, float* __restrict__ __ins_36, int8_t* __restrict__ __ins_37, float* __restrict__ __ins_38, float* __restrict__ __ins_39, int8_t* __restrict__ __ins_40, float* __restrict__ __ins_41, float* __restrict__ __ins_42, int8_t* __restrict__ __ins_43, float* __restrict__ __ins_44, float* __restrict__ __ins_45, int8_t* __restrict__ __ins_46, float* __restrict__ __ins_47, float* __restrict__ __ins_48, int8_t* __restrict__ __ins_49, float* __restrict__ __ins_50, float* __restrict__ __ins_51, int8_t* __restrict__ __ins_52, float* __restrict__ __ins_53, float* __restrict__ __ins_54, int8_t* __restrict__ __ins_55, float* __restrict__ __ins_56, float* __restrict__ __ins_57, int8_t* __restrict__ __ins_58, float* __restrict__ __ins_59, float* __restrict__ __ins_60, int8_t* __restrict__ __ins_61, float* __restrict__ __ins_62, float* __restrict__ __ins_63, int8_t* __restrict__ __ins_64, float* __restrict__ __ins_65, float* __restrict__ __ins_66, int8_t* __restrict__ __ins_67, float* __restrict__ __ins_68, float* __restrict__ __ins_69) noexcept{
  for (uint64_t __batchwise_iter_0 = 0UL; __batchwise_iter_0 < 8UL; __batchwise_iter_0 += 1UL) {
    int8_t* __rescheduled_1 = (int8_t*)sc_aligned_malloc(__stream, 2021632UL);
    int8_t* conv_out_pointer = (int8_t*)&__rescheduled_1[0UL];
    int8_t* pool_out_pointer = (int8_t*)&__rescheduled_1[1218816UL];
    void* scratchpad_pointer = (void*)&__rescheduled_1[1218816UL];
    dnnl::memory conv_weights_memory = dnnl::memory({{conv_weights_tz_stg1_}, dt::s8, tag::Adcb16a}, eng_dnn_, conv_weights_ptr_stg1_);
    dnnl::memory conv_bias_memory = dnnl::memory({{conv_bias_tz_stg1_}, dt::f32, tag::a}, eng_dnn_, conv_bias_ptr_stg1_);

    runStart(reinterpret_cast<int8_t*>(input_pointers[__batchwise_iter_0]), conv_out_pointer, pool_out_pointer, scratchpad_pointer);
    // [s8 [1, 1, 1, 58, 58, 64] @ A1aBCD64b]
    int8_t* buffer_71 = (int8_t*)&__rescheduled_1[802816UL];
    res2a_conv_0_cast_mul_add_cast_relu__8(buffer_71, pool_out_pointer, &__ins_4[0UL], &__ins_5[0UL], &__ins_6[0UL]);
    // [s8 [1, 1, 1, 56, 56, 64] @ A1aBCD64b]
 
    int8_t* buffer_70 = (int8_t*)&__rescheduled_1[0UL];
    res2a_conv_b_cast_mul_add_cast__4(buffer_70, pool_out_pointer, &__ins_1[0UL], &__ins_2[0UL], &__ins_3[0UL]);
    // [s8 [1, 1, 1, 58, 58, 64] @ A1aBCD64b]
     int8_t* buffer_72 = (int8_t*)&__rescheduled_1[1233408UL];
    res2a_conv_1_cast_mul_add_cast_relu__12(buffer_72, buffer_71, &__ins_7[0UL], &__ins_8[0UL], &__ins_9[0UL]);
  
  
    res2a_conv_2_cast_mul_add_cast_add_relu__16(buffer_70, buffer_72, &__ins_10[0UL], &__ins_11[0UL], &__ins_12[0UL], buffer_70);
    res2b_conv_0_cast_mul_add_cast_relu__20(buffer_72, buffer_70, &__ins_13[0UL], &__ins_14[0UL], &__ins_15[0UL]);
    res2b_conv_1_cast_mul_add_cast_relu__24(buffer_71, buffer_72, &__ins_16[0UL], &__ins_17[0UL], &__ins_18[0UL]);
    res2b_conv_2_cast_mul_add_cast_add_relu__28(buffer_70, buffer_71, &__ins_19[0UL], &__ins_20[0UL], &__ins_21[0UL], buffer_70);
    res2c_conv_0_cast_mul_add_cast_relu__32(buffer_72, buffer_70, &__ins_22[0UL], &__ins_23[0UL], &__ins_24[0UL]);
    res2c_conv_1_cast_mul_add_cast_relu__36(buffer_71, buffer_72, &__ins_25[0UL], &__ins_26[0UL], &__ins_27[0UL]);
    res2c_conv_2_cast_mul_add_cast_add_relu__40(buffer_70, buffer_71, &__ins_28[0UL], &__ins_29[0UL], &__ins_30[0UL], buffer_70);
    res3a_conv_b_cast_mul_add_cast__44(buffer_72, buffer_70, &__ins_31[0UL], &__ins_32[0UL], &__ins_33[0UL]);
    res3a_conv_0_cast_mul_add_cast_relu__48(buffer_71, buffer_70, &__ins_34[0UL], &__ins_35[0UL], &__ins_36[0UL]);
    res3a_conv_1_cast_mul_add_cast_relu__52(buffer_70, buffer_71, &__ins_37[0UL], &__ins_38[0UL], &__ins_39[0UL]);
    res3a_conv_2_cast_mul_add_cast_add_relu__56(buffer_72, buffer_70, &__ins_40[0UL], &__ins_41[0UL], &__ins_42[0UL], buffer_72);
    res3b_conv_0_cast_mul_add_cast_relu__60(buffer_70, buffer_72, &__ins_43[0UL], &__ins_44[0UL], &__ins_45[0UL]);
    res3b_conv_1_cast_mul_add_cast_relu__64(buffer_71, buffer_70, &__ins_46[0UL], &__ins_47[0UL], &__ins_48[0UL]);
    res3b_conv_2_cast_mul_add_cast_add_relu__68(buffer_72, buffer_71, &__ins_49[0UL], &__ins_50[0UL], &__ins_51[0UL], buffer_72);
    res3c_conv_0_cast_mul_add_cast_relu__72(buffer_70, buffer_72, &__ins_52[0UL], &__ins_53[0UL], &__ins_54[0UL]);
    res3c_conv_1_cast_mul_add_cast_relu__76(buffer_71, buffer_70, &__ins_55[0UL], &__ins_56[0UL], &__ins_57[0UL]);
    res3c_conv_2_cast_mul_add_cast_add_relu__80(buffer_72, buffer_71, &__ins_58[0UL], &__ins_59[0UL], &__ins_60[0UL], buffer_72);
    res3d_conv_0_cast_mul_add_cast_relu__84(buffer_70, buffer_72, &__ins_61[0UL], &__ins_62[0UL], &__ins_63[0UL]);
    res3d_conv_1_cast_mul_add_cast_relu__88(buffer_71, buffer_70, &__ins_64[0UL], &__ins_65[0UL], &__ins_66[0UL]);
    res3d_conv_2_cast_mul_add_cast_add_relu__93(&__outs_0[(__batchwise_iter_0 * 401408UL)], buffer_71, &__ins_67[0UL], &__ins_68[0UL], &__ins_69[0UL], buffer_72);
    sc_aligned_free(__stream, __rescheduled_1);
  }
  return true;
}

extern "C" void rn50_backbone_wrapper_bs8(int8_t* __restrict__ backbone_output,int64_t* __restrict__ input_pointers,float* __restrict__ final_out, float* __restrict__ res2a_weight_b, float* __restrict__ res2a_bias_b, float* __restrict__ res2a_weight_0, float* __restrict__ res2a_bias_0, float* __restrict__ res2a_weight_1, float* __restrict__ res2a_bias_1, float* __restrict__ res2a_weight_2, float* __restrict__ res2a_bias_2, float* __restrict__ res2b_weight_0, float* __restrict__ res2b_bias_0, float* __restrict__ res2b_weight_1, float* __restrict__ res2b_bias_1, float* __restrict__ res2b_weight_2, float* __restrict__ res2b_bias_2, float* __restrict__ res2c_weight_0, float* __restrict__ res2c_bias_0, float* __restrict__ res2c_weight_1, float* __restrict__ res2c_bias_1, float* __restrict__ res2c_weight_2, float* __restrict__ res2c_bias_2, float* __restrict__ res3a_weight_b, float* __restrict__ res3a_bias_b, float* __restrict__ res3a_weight_0, float* __restrict__ res3a_bias_0, float* __restrict__ res3a_weight_1, float* __restrict__ res3a_bias_1, float* __restrict__ res3a_weight_2, float* __restrict__ res3a_bias_2, float* __restrict__ res3b_weight_0, float* __restrict__ res3b_bias_0, float* __restrict__ res3b_weight_1, float* __restrict__ res3b_bias_1, float* __restrict__ res3b_weight_2, float* __restrict__ res3b_bias_2, float* __restrict__ res3c_weight_0, float* __restrict__ res3c_bias_0, float* __restrict__ res3c_weight_1, float* __restrict__ res3c_bias_1, float* __restrict__ res3c_weight_2, float* __restrict__ res3c_bias_2, float* __restrict__ res3d_weight_0, float* __restrict__ res3d_bias_0, float* __restrict__ res3d_weight_1, float* __restrict__ res3d_bias_1, float* __restrict__ res3d_weight_2, float* __restrict__ res3d_bias_2, float* __restrict__ res4a_weight_b, float* __restrict__ res4a_bias_b, float* __restrict__ res4a_weight_0, float* __restrict__ res4a_bias_0, float* __restrict__ res4a_weight_1, float* __restrict__ res4a_bias_1, float* __restrict__ res4a_weight_2, float* __restrict__ res4a_bias_2, float* __restrict__ res4b_weight_0, float* __restrict__ res4b_bias_0, float* __restrict__ res4b_weight_1, float* __restrict__ res4b_bias_1, float* __restrict__ res4b_weight_2, float* __restrict__ res4b_bias_2, float* __restrict__ res4c_weight_0, float* __restrict__ res4c_bias_0, float* __restrict__ res4c_weight_1, float* __restrict__ res4c_bias_1, float* __restrict__ res4c_weight_2, float* __restrict__ res4c_bias_2, float* __restrict__ res4d_weight_0, float* __restrict__ res4d_bias_0, float* __restrict__ res4d_weight_1, float* __restrict__ res4d_bias_1, float* __restrict__ res4d_weight_2, float* __restrict__ res4d_bias_2, float* __restrict__ res4e_weight_0, float* __restrict__ res4e_bias_0, float* __restrict__ res4e_weight_1, float* __restrict__ res4e_bias_1, float* __restrict__ res4e_weight_2, float* __restrict__ res4e_bias_2, float* __restrict__ res4f_weight_0, float* __restrict__ res4f_bias_0, float* __restrict__ res4f_weight_1, float* __restrict__ res4f_bias_1, float* __restrict__ res4f_weight_2, float* __restrict__ res4f_bias_2, float* __restrict__ res5a_weight_b, float* __restrict__ res5a_bias_b, float* __restrict__ res5a_weight_0, float* __restrict__ res5a_bias_0, float* __restrict__ res5a_weight_1, float* __restrict__ res5a_bias_1, float* __restrict__ res5a_weight_2, float* __restrict__ res5a_bias_2, float* __restrict__ res5b_weight_0, float* __restrict__ res5b_bias_0, float* __restrict__ res5b_weight_1, float* __restrict__ res5b_bias_1, float* __restrict__ res5b_weight_2, float* __restrict__ res5b_bias_2, float* __restrict__ res5c_weight_0, float* __restrict__ res5c_bias_0, float* __restrict__ res5c_weight_1, float* __restrict__ res5c_bias_1, float* __restrict__ res5c_weight_2, float* __restrict__ res5c_bias_2) noexcept{
 bool& is_init = *(bool*)(__module_data + 0);
  int8_t* folded_const_261 = (int8_t*)&__uninitialized_data[216064UL];
  float* folded_const_156 = (float*)&__uninitialized_data[0UL];
  float* folded_const_222 = (float*)&__uninitialized_data[111616UL];
  int8_t* folded_const_260 = (int8_t*)&__uninitialized_data[211968UL];
  float* folded_const_157 = (float*)&__uninitialized_data[1024UL];
  float* folded_const_208 = (float*)&__uninitialized_data[105984UL];
  int8_t* folded_const_268 = (int8_t*)&__uninitialized_data[347136UL];
  float* folded_const_158 = (float*)&__uninitialized_data[1280UL];
  float* folded_const_209 = (float*)&__uninitialized_data[106240UL];
  int8_t* folded_const_262 = (int8_t*)&__uninitialized_data[232448UL];
  float* folded_const_159 = (float*)&__uninitialized_data[1536UL];
  float* folded_const_223 = (float*)&__uninitialized_data[112640UL];
  int8_t* folded_const_265 = (int8_t*)&__uninitialized_data[281600UL];
  float* folded_const_160 = (float*)&__uninitialized_data[2560UL];
  float* folded_const_210 = (float*)&__uninitialized_data[106496UL];
  int8_t* folded_const_269 = (int8_t*)&__uninitialized_data[384000UL];
  float* folded_const_161 = (float*)&__uninitialized_data[2816UL];
  float* folded_const_211 = (float*)&__uninitialized_data[106752UL];
  int8_t* folded_const_263 = (int8_t*)&__uninitialized_data[248832UL];
  float* folded_const_162 = (float*)&__uninitialized_data[3072UL];
  float* folded_const_224 = (float*)&__uninitialized_data[113664UL];
  int8_t* folded_const_266 = (int8_t*)&__uninitialized_data[297984UL];
  float* folded_const_163 = (float*)&__uninitialized_data[4096UL];
  float* folded_const_212 = (float*)&__uninitialized_data[107008UL];
  int8_t* folded_const_270 = (int8_t*)&__uninitialized_data[420864UL];
  float* folded_const_164 = (float*)&__uninitialized_data[4352UL];
  float* folded_const_213 = (float*)&__uninitialized_data[107264UL];
  int8_t* folded_const_264 = (int8_t*)&__uninitialized_data[265216UL];
  float* folded_const_165 = (float*)&__uninitialized_data[4608UL];
  float* folded_const_225 = (float*)&__uninitialized_data[114688UL];
  int8_t* folded_const_278 = (int8_t*)&__uninitialized_data[916480UL];
  float* folded_const_166 = (float*)&__uninitialized_data[5632UL];
  float* folded_const_238 = (float*)&__uninitialized_data[128000UL];
  int8_t* folded_const_267 = (int8_t*)&__uninitialized_data[314368UL];
  float* folded_const_167 = (float*)&__uninitialized_data[7680UL];
  float* folded_const_214 = (float*)&__uninitialized_data[107520UL];
  int8_t* folded_const_280 = (int8_t*)&__uninitialized_data[1178624UL];
  float* folded_const_168 = (float*)&__uninitialized_data[8192UL];
  float* folded_const_215 = (float*)&__uninitialized_data[108032UL];
  int8_t* folded_const_271 = (int8_t*)&__uninitialized_data[457728UL];
  float* folded_const_169 = (float*)&__uninitialized_data[8704UL];
  float* folded_const_239 = (float*)&__uninitialized_data[130048UL];
  int8_t* folded_const_275 = (int8_t*)&__uninitialized_data[719872UL];
  float* folded_const_170 = (float*)&__uninitialized_data[10752UL];
  float* folded_const_216 = (float*)&__uninitialized_data[108544UL];
  int8_t* folded_const_281 = (int8_t*)&__uninitialized_data[1326080UL];
  float* folded_const_171 = (float*)&__uninitialized_data[11264UL];
  float* folded_const_217 = (float*)&__uninitialized_data[109056UL];
  int8_t* folded_const_272 = (int8_t*)&__uninitialized_data[523264UL];
  float* folded_const_172 = (float*)&__uninitialized_data[11776UL];
  float* folded_const_240 = (float*)&__uninitialized_data[132096UL];
  int8_t* folded_const_276 = (int8_t*)&__uninitialized_data[785408UL];
  float* folded_const_173 = (float*)&__uninitialized_data[13824UL];
  float* folded_const_218 = (float*)&__uninitialized_data[109568UL];
  int8_t* folded_const_282 = (int8_t*)&__uninitialized_data[1473536UL];
  float* folded_const_174 = (float*)&__uninitialized_data[14336UL];
  float* folded_const_219 = (float*)&__uninitialized_data[110080UL];
  int8_t* folded_const_273 = (int8_t*)&__uninitialized_data[588800UL];
  float* folded_const_175 = (float*)&__uninitialized_data[14848UL];
  float* folded_const_241 = (float*)&__uninitialized_data[134144UL];
  int8_t* folded_const_277 = (int8_t*)&__uninitialized_data[850944UL];
  float* folded_const_176 = (float*)&__uninitialized_data[16896UL];
  float* folded_const_220 = (float*)&__uninitialized_data[110592UL];
  int8_t* folded_const_283 = (int8_t*)&__uninitialized_data[1620992UL];
  float* folded_const_177 = (float*)&__uninitialized_data[17408UL];
  float* folded_const_221 = (float*)&__uninitialized_data[111104UL];
  int8_t* folded_const_274 = (int8_t*)&__uninitialized_data[654336UL];
  float* folded_const_178 = (float*)&__uninitialized_data[17920UL];
  float* folded_const_242 = (float*)&__uninitialized_data[136192UL];
  int8_t* folded_const_295 = (int8_t*)&__uninitialized_data[4652032UL];
  float* folded_const_179 = (float*)&__uninitialized_data[19968UL];
  float* folded_const_249 = (float*)&__uninitialized_data[150528UL];
  int8_t* folded_const_279 = (int8_t*)&__uninitialized_data[1047552UL];
  float* folded_const_180 = (float*)&__uninitialized_data[24064UL];
  float* folded_const_226 = (float*)&__uninitialized_data[115712UL];
  int8_t* folded_const_297 = (int8_t*)&__uninitialized_data[5700608UL];
  float* folded_const_181 = (float*)&__uninitialized_data[25088UL];
  float* folded_const_227 = (float*)&__uninitialized_data[116736UL];
  int8_t* folded_const_284 = (int8_t*)&__uninitialized_data[1768448UL];
  float* folded_const_182 = (float*)&__uninitialized_data[26112UL];
  float* folded_const_250 = (float*)&__uninitialized_data[154624UL];
  int8_t* folded_const_290 = (int8_t*)&__uninitialized_data[3341312UL];
  float* folded_const_183 = (float*)&__uninitialized_data[30208UL];
  float* folded_const_228 = (float*)&__uninitialized_data[117760UL];
  int8_t* folded_const_298 = (int8_t*)&__uninitialized_data[6290432UL];
  float* folded_const_184 = (float*)&__uninitialized_data[31232UL];
  float* folded_const_229 = (float*)&__uninitialized_data[118784UL];
  int8_t* folded_const_285 = (int8_t*)&__uninitialized_data[2030592UL];
  float* folded_const_185 = (float*)&__uninitialized_data[32256UL];
  float* folded_const_251 = (float*)&__uninitialized_data[158720UL];
  int8_t* folded_const_291 = (int8_t*)&__uninitialized_data[3603456UL];
  float* folded_const_186 = (float*)&__uninitialized_data[36352UL];
  float* folded_const_230 = (float*)&__uninitialized_data[119808UL];
  int8_t* folded_const_299 = (int8_t*)&__uninitialized_data[6880256UL];
  float* folded_const_187 = (float*)&__uninitialized_data[37376UL];
  float* folded_const_231 = (float*)&__uninitialized_data[120832UL];
  int8_t* folded_const_286 = (int8_t*)&__uninitialized_data[2292736UL];
  float* folded_const_188 = (float*)&__uninitialized_data[38400UL];
  float* folded_const_252 = (float*)&__uninitialized_data[162816UL];
  int8_t* folded_const_292 = (int8_t*)&__uninitialized_data[3865600UL];
  float* folded_const_189 = (float*)&__uninitialized_data[42496UL];
  float* folded_const_232 = (float*)&__uninitialized_data[121856UL];
  int8_t* folded_const_300 = (int8_t*)&__uninitialized_data[7470080UL];
  float* folded_const_190 = (float*)&__uninitialized_data[43520UL];
  float* folded_const_233 = (float*)&__uninitialized_data[122880UL];
  int8_t* folded_const_287 = (int8_t*)&__uninitialized_data[2554880UL];
  float* folded_const_191 = (float*)&__uninitialized_data[44544UL];
  float* folded_const_253 = (float*)&__uninitialized_data[166912UL];
  int8_t* folded_const_293 = (int8_t*)&__uninitialized_data[4127744UL];
  float* folded_const_192 = (float*)&__uninitialized_data[48640UL];
  float* folded_const_234 = (float*)&__uninitialized_data[123904UL];
  int8_t* folded_const_301 = (int8_t*)&__uninitialized_data[8059904UL];
  float* folded_const_193 = (float*)&__uninitialized_data[49664UL];
  float* folded_const_235 = (float*)&__uninitialized_data[124928UL];
  int8_t* folded_const_288 = (int8_t*)&__uninitialized_data[2817024UL];
  float* folded_const_194 = (float*)&__uninitialized_data[50688UL];
  float* folded_const_254 = (float*)&__uninitialized_data[171008UL];
  int8_t* folded_const_294 = (int8_t*)&__uninitialized_data[4389888UL];
  float* folded_const_195 = (float*)&__uninitialized_data[54784UL];
  float* folded_const_236 = (float*)&__uninitialized_data[125952UL];
  int8_t* folded_const_302 = (int8_t*)&__uninitialized_data[8649728UL];
  float* folded_const_196 = (float*)&__uninitialized_data[55808UL];
  float* folded_const_237 = (float*)&__uninitialized_data[126976UL];
  int8_t* folded_const_289 = (int8_t*)&__uninitialized_data[3079168UL];
  float* folded_const_197 = (float*)&__uninitialized_data[56832UL];
  float* folded_const_255 = (float*)&__uninitialized_data[175104UL];
  int8_t* folded_const_308 = (int8_t*)&__uninitialized_data[14482432UL];
  float* folded_const_198 = (float*)&__uninitialized_data[60928UL];
  float* folded_const_256 = (float*)&__uninitialized_data[179200UL];
  int8_t* folded_const_296 = (int8_t*)&__uninitialized_data[5176320UL];
  float* folded_const_199 = (float*)&__uninitialized_data[69120UL];
  float* folded_const_243 = (float*)&__uninitialized_data[138240UL];
  int8_t* folded_const_309 = (int8_t*)&__uninitialized_data[16579584UL];
  float* folded_const_200 = (float*)&__uninitialized_data[71168UL];
  float* folded_const_244 = (float*)&__uninitialized_data[140288UL];
  int8_t* folded_const_303 = (int8_t*)&__uninitialized_data[9239552UL];
  float* folded_const_201 = (float*)&__uninitialized_data[73216UL];
  float* folded_const_257 = (float*)&__uninitialized_data[187392UL];
  int8_t* folded_const_306 = (int8_t*)&__uninitialized_data[12385280UL];
  float* folded_const_202 = (float*)&__uninitialized_data[81408UL];
  float* folded_const_245 = (float*)&__uninitialized_data[142336UL];
  int8_t* folded_const_310 = (int8_t*)&__uninitialized_data[18938880UL];
  float* folded_const_203 = (float*)&__uninitialized_data[83456UL];
  float* folded_const_246 = (float*)&__uninitialized_data[144384UL];
  int8_t* folded_const_304 = (int8_t*)&__uninitialized_data[10288128UL];
  float* folded_const_204 = (float*)&__uninitialized_data[85504UL];
  float* folded_const_258 = (float*)&__uninitialized_data[195584UL];
  int8_t* folded_const_307 = (int8_t*)&__uninitialized_data[13433856UL];
  float* folded_const_205 = (float*)&__uninitialized_data[93696UL];
  float* folded_const_247 = (float*)&__uninitialized_data[146432UL];
  int8_t* folded_const_311 = (int8_t*)&__uninitialized_data[21298176UL];
  float* folded_const_206 = (float*)&__uninitialized_data[95744UL];
  float* folded_const_248 = (float*)&__uninitialized_data[148480UL];
  int8_t* folded_const_305 = (int8_t*)&__uninitialized_data[11336704UL];
  float* folded_const_207 = (float*)&__uninitialized_data[97792UL];
  float* folded_const_259 = (float*)&__uninitialized_data[203776UL];
  int8_t* __rescheduled_0 = (int8_t*)sc_aligned_malloc(__stream, 86638592UL);
  if (!is_init) {
    __init_const_globals(backbone_output, reinterpret_cast<int8_t*>(input_pointers), res2a_weight_b, res2a_bias_b, res2a_weight_0, res2a_bias_0, res2a_weight_1, res2a_bias_1, res2a_weight_2, res2a_bias_2, res2b_weight_0, res2b_bias_0, res2b_weight_1, res2b_bias_1, res2b_weight_2, res2b_bias_2, res2c_weight_0, res2c_bias_0, res2c_weight_1, res2c_bias_1, res2c_weight_2, res2c_bias_2, res3a_weight_b, res3a_bias_b, res3a_weight_0, res3a_bias_0, res3a_weight_1, res3a_bias_1, res3a_weight_2, res3a_bias_2, res3b_weight_0, res3b_bias_0, res3b_weight_1, res3b_bias_1, res3b_weight_2, res3b_bias_2, res3c_weight_0, res3c_bias_0, res3c_weight_1, res3c_bias_1, res3c_weight_2, res3c_bias_2, res3d_weight_0, res3d_bias_0, res3d_weight_1, res3d_bias_1, res3d_weight_2, res3d_bias_2, res4a_weight_b, res4a_bias_b, res4a_weight_0, res4a_bias_0, res4a_weight_1, res4a_bias_1, res4a_weight_2, res4a_bias_2, res4b_weight_0, res4b_bias_0, res4b_weight_1, res4b_bias_1, res4b_weight_2, res4b_bias_2, res4c_weight_0, res4c_bias_0, res4c_weight_1, res4c_bias_1, res4c_weight_2, res4c_bias_2, res4d_weight_0, res4d_bias_0, res4d_weight_1, res4d_bias_1, res4d_weight_2, res4d_bias_2, res4e_weight_0, res4e_bias_0, res4e_weight_1, res4e_bias_1, res4e_weight_2, res4e_bias_2, res4f_weight_0, res4f_bias_0, res4f_weight_1, res4f_bias_1, res4f_weight_2, res4f_bias_2, res5a_weight_b, res5a_bias_b, res5a_weight_0, res5a_bias_0, res5a_weight_1, res5a_bias_1, res5a_weight_2, res5a_bias_2, res5b_weight_0, res5b_bias_0, res5b_weight_1, res5b_bias_1, res5b_weight_2, res5b_bias_2, res5c_weight_0, res5c_bias_0, res5c_weight_1, res5c_bias_1, res5c_weight_2, res5c_bias_2);
  }
  // [s8 [8, 1, 8, 28, 28, 64] @ A1aBCD64b]
  int8_t* buffer_611 = (int8_t*)&__rescheduled_0[0UL];
  mlperf_batchwise_8_fused_res2a_conv_b_cast_mul_add_cast_reorder_res2a_conv_0_cast_mul_add_relu_cast_res2a_conv_1_cast_mul_add_relu_cast_reorder_res2a_conv_2_cast_mul_add_cast_add_cast_res2b_conv_0_cast_mul_add_relu_cast_res2b_conv_1_cast_mul_add_relu_cast_reorder_res2b_conv_2_cast_mul_add_cast_add_cast_res2c_conv_0_cast_mul_add_relu_cast_res2c_conv_1_cast_mul_add_relu_cast_reorder_res2c_conv_2_cast_mul_add_cast_add_cast_reorder_res3a_conv_b_cast_mul_add_cast_res3a_conv_0_cast_mul_add_relu_cast_res3a_conv_1_cast_mul_add_relu_cast_res3a_conv_2_cast_mul_add_cast_add_cast_reorder_res3b_conv_0_cast_mul_add_relu_cast_reorder_res3b_conv_1_cast_mul_add_relu_cast_reorder_res3b_conv_2_cast_mul_add_cast_add_cast_reorder_res3c_conv_0_cast_mul_add_relu_cast_res3c_conv_1_cast_mul_add_relu_cast_reorder_res3c_conv_2_cast_mul_add_cast_add_cast_reorder_res3d_conv_0_cast_mul_add_relu_cast_reorder_res3d_conv_1_cast_mul_add_relu_cast_reorder_res3d_conv_2_cast_mul_add_cast_add_cast__685(buffer_611, input_pointers, folded_const_261, folded_const_156, folded_const_222, folded_const_260, folded_const_157, folded_const_208, folded_const_268, folded_const_158, folded_const_209, folded_const_262, folded_const_159, folded_const_223, folded_const_265, folded_const_160, folded_const_210, folded_const_269, folded_const_161, folded_const_211, folded_const_263, folded_const_162, folded_const_224, folded_const_266, folded_const_163, folded_const_212, folded_const_270, folded_const_164, folded_const_213, folded_const_264, folded_const_165, folded_const_225, folded_const_278, folded_const_166, folded_const_238, folded_const_267, folded_const_167, folded_const_214, folded_const_280, folded_const_168, folded_const_215, folded_const_271, folded_const_169, folded_const_239, folded_const_275, folded_const_170, folded_const_216, folded_const_281, folded_const_171, folded_const_217, folded_const_272, folded_const_172, folded_const_240, folded_const_276, folded_const_173, folded_const_218, folded_const_282, folded_const_174, folded_const_219, folded_const_273, folded_const_175, folded_const_241, folded_const_277, folded_const_176, folded_const_220, folded_const_283, folded_const_177, folded_const_221, folded_const_274, folded_const_178, folded_const_242);
  // [s8 [4, 2, 16, 14, 14, 64] @ A2aBCD64b]
  int8_t* buffer_612 = (int8_t*)&__rescheduled_0[42467328UL];
  batchwise_4_fused_res4a_conv_b_cast_mul_add_cast_res4a_conv_0_cast_mul_add_cast_relu_res4a_conv_1_cast_mul_add_cast_relu_res4a_conv_2_cast_mul_add_cast_add_relu_res4b_conv_0_cast_mul_add_cast_relu_res4b_conv_1_cast_mul_add_cast_relu_res4b_conv_2_cast_mul_add_cast_add_relu_res4c_conv_0_cast_mul_add_cast_relu_res4c_conv_1_cast_mul_add_cast_relu_res4c_conv_2_cast_mul_add_cast_add_relu_res4d_conv_0_cast_mul_add_cast_relu_res4d_conv_1_cast_mul_add_cast_relu_res4d_conv_2_cast_mul_add_cast_add_relu_res4e_conv_0_cast_mul_add_cast_relu_res4e_conv_1_cast_mul_add_cast_relu_res4e_conv_2_cast_mul_add_cast_add_relu_res4f_conv_0_cast_mul_add_cast_relu_res4f_conv_1_cast_mul_add_cast_relu_res4f_conv_2_cast_mul_add_cast_add_relu__685(buffer_612, &buffer_611[0UL], folded_const_295, folded_const_179, folded_const_249, folded_const_279, folded_const_180, folded_const_226, folded_const_297, folded_const_181, folded_const_227, folded_const_284, folded_const_182, folded_const_250, folded_const_290, folded_const_183, folded_const_228, folded_const_298, folded_const_184, folded_const_229, folded_const_285, folded_const_185, folded_const_251, folded_const_291, folded_const_186, folded_const_230, folded_const_299, folded_const_187, folded_const_231, folded_const_286, folded_const_188, folded_const_252, folded_const_292, folded_const_189, folded_const_232, folded_const_300, folded_const_190, folded_const_233, folded_const_287, folded_const_191, folded_const_253, folded_const_293, folded_const_192, folded_const_234, folded_const_301, folded_const_193, folded_const_235, folded_const_288, folded_const_194, folded_const_254, folded_const_294, folded_const_195, folded_const_236, folded_const_302, folded_const_196, folded_const_237, folded_const_289, folded_const_197, folded_const_255);
  // [s8 [8, 1, 4, 7, 7, 512] @ A1aBCD512b]
  int8_t* buffer_613 = (int8_t*)&__rescheduled_0[0UL];
  res5a_conv_b_cast_mul_add_cast__683(buffer_613, &buffer_612[0UL], folded_const_308, folded_const_198, folded_const_256);
  // [s8 [8, 1, 8, 16, 16, 64] @ A1aBCD64b]
  int8_t* buffer_614 = (int8_t*)&__rescheduled_0[53084160UL];
  res5a_conv_0_cast_mul_add_cast_relu_reorder__682(buffer_614, &buffer_612[0UL], folded_const_296, folded_const_199, folded_const_243);
  // [s8 [8, 1, 1, 7, 7, 512] @ A1aBCD512b]
  int8_t* buffer_615 = (int8_t*)&__rescheduled_0[42467328UL];
  res5a_conv_1_cast_mul_add_cast_relu_reorder__681(buffer_615, buffer_614, folded_const_309, folded_const_200, folded_const_244);
  res5a_conv_2_cast_mul_add_cast_add_relu__680(buffer_613, buffer_615, folded_const_303, folded_const_201, folded_const_257, buffer_613);
  res5b_conv_0_cast_mul_add_cast_relu__679(buffer_615, buffer_613, folded_const_306, folded_const_202, folded_const_245);
  res5b_conv_1_cast_mul_add_cast_relu_reorder__678(buffer_614, buffer_615, folded_const_310, folded_const_203, folded_const_246);
  res5b_conv_2_cast_mul_add_cast_add_relu__677(buffer_613, buffer_614, folded_const_304, folded_const_204, folded_const_258, buffer_613);
  res5c_conv_0_cast_mul_add_cast_relu_reorder__676(buffer_615, buffer_613, folded_const_307, folded_const_205, folded_const_247);
  res5c_conv_1_cast_mul_add_cast_relu__675(buffer_614, buffer_615, folded_const_311, folded_const_206, folded_const_248);
   int8_t* avg_pool_in = (int8_t*)&__rescheduled_0[1000000UL];
  res5c_conv_2_cast_mul_add_cast_add_relu_reorder__674(avg_pool_in, buffer_614, folded_const_305, folded_const_207, folded_const_259, buffer_613);


  int8_t* avg_pool_out = (int8_t*)&__rescheduled_0[1802816UL];
  
  int8_t* fc_src_mem = (int8_t*)&__rescheduled_0[0UL];

  void* fc_scratch_ptr = (int8_t*)&__rescheduled_0[3387391UL];

  runEnd(avg_pool_in,avg_pool_out,final_out,fc_scratch_ptr,fc_src_mem);
  
  sc_aligned_free(__stream, __rescheduled_0);
}
 